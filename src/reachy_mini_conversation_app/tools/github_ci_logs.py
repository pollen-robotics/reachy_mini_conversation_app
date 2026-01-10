"""GitHub CI logs tool - get workflow run logs using PyGithub."""

import io
import logging
import zipfile
from typing import Any, Dict, List, Optional

import requests
from github import Github, GithubException

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class GitHubCILogsTool(Tool):
    """Get CI/CD workflow logs from GitHub Actions."""

    name = "github_ci_logs"
    description = (
        "Get the logs from GitHub Actions workflow runs. "
        "Can retrieve logs for a specific job or all jobs in a workflow run. "
        "Useful for debugging CI failures by examining the actual output."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository name (e.g., 'owner/repo' or just 'repo' if GITHUB_DEFAULT_OWNER is set)",
            },
            "run_id": {
                "type": "integer",
                "description": "Workflow run ID (get from github_pr_checks or GitHub URL)",
            },
            "job_name": {
                "type": "string",
                "description": "Specific job name to get logs for (optional, gets all jobs if not specified)",
            },
            "failed_only": {
                "type": "boolean",
                "description": "Only return logs from failed jobs (default: false)",
            },
            "tail_lines": {
                "type": "integer",
                "description": "Number of lines to return from end of each log (default: 100, max: 500)",
            },
        },
        "required": ["repo", "run_id"],
    }

    def _get_full_repo_name(self, repo: str) -> str:
        """Get full repo name with owner."""
        if "/" in repo:
            return repo
        if config.GITHUB_DEFAULT_OWNER:
            return f"{config.GITHUB_DEFAULT_OWNER}/{repo}"
        raise ValueError(
            f"Repository '{repo}' must include owner (e.g., 'owner/repo') "
            "or set GITHUB_DEFAULT_OWNER in environment."
        )

    def _extract_logs_from_zip(
        self,
        zip_content: bytes,
        job_name: Optional[str] = None,
        failed_jobs: Optional[List[str]] = None,
        tail_lines: int = 100,
    ) -> Dict[str, str]:
        """Extract logs from the downloaded zip file."""
        logs = {}
        failed_jobs = failed_jobs or []

        try:
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                for name in zf.namelist():
                    # Log files are typically named like "job_name/step_name.txt"
                    if name.endswith(".txt"):
                        # Extract job name from path
                        parts = name.split("/")
                        current_job = parts[0] if len(parts) > 1 else name

                        # Filter by job name if specified
                        if job_name and job_name.lower() not in current_job.lower():
                            continue

                        # Read the log content
                        content = zf.read(name).decode("utf-8", errors="replace")

                        # Get last N lines if specified
                        if tail_lines > 0:
                            lines = content.splitlines()
                            content = "\n".join(lines[-tail_lines:])

                        # Add to logs dict
                        if current_job not in logs:
                            logs[current_job] = ""
                        logs[current_job] += f"\n--- {name} ---\n{content}\n"

        except zipfile.BadZipFile:
            logger.error("Failed to parse logs zip file")
            return {"error": "Failed to parse logs archive"}

        return logs

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Get CI workflow logs from GitHub Actions."""
        repo_name = kwargs.get("repo", "")
        run_id = kwargs.get("run_id")
        job_name = kwargs.get("job_name")
        failed_only = kwargs.get("failed_only", False)
        tail_lines = min(kwargs.get("tail_lines", 100), 500)

        logger.info(f"Tool call: github_ci_logs - repo='{repo_name}', run_id={run_id}")

        if not repo_name:
            return {"error": "Repository name is required."}
        if not run_id:
            return {"error": "Workflow run ID is required."}

        # Check for token
        token = config.GITHUB_TOKEN
        if not token:
            return {"error": "GITHUB_TOKEN is required to get CI logs."}

        try:
            full_repo = self._get_full_repo_name(repo_name)
        except ValueError as e:
            return {"error": str(e)}

        try:
            g = Github(token)
            gh_repo = g.get_repo(full_repo)

            # Get the workflow run
            try:
                run = gh_repo.get_workflow_run(run_id)
            except GithubException as e:
                if e.status == 404:
                    return {"error": f"Workflow run {run_id} not found."}
                raise

            # Get jobs for this run
            jobs = run.jobs()
            job_info: List[Dict[str, Any]] = []
            failed_jobs: List[str] = []

            for job in jobs:
                info = {
                    "id": job.id,
                    "name": job.name,
                    "status": job.status,
                    "conclusion": job.conclusion,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                }

                # Track failed jobs
                if job.conclusion in ["failure", "timed_out"]:
                    failed_jobs.append(job.name)

                # Get steps info
                steps = []
                for step in job.steps:
                    step_info = {
                        "name": step.name,
                        "status": step.status,
                        "conclusion": step.conclusion,
                        "number": step.number,
                    }
                    steps.append(step_info)
                info["steps"] = steps

                job_info.append(info)

            # Filter jobs if failed_only
            if failed_only:
                job_info = [j for j in job_info if j["conclusion"] in ["failure", "timed_out"]]
                if not job_info:
                    return {
                        "status": "success",
                        "message": "No failed jobs found in this workflow run.",
                        "run_id": run_id,
                        "workflow": run.name,
                        "conclusion": run.conclusion,
                    }

            # Download logs using the GitHub API
            logs_url = run.logs_url

            # Use requests to download the logs (PyGithub doesn't have direct support)
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.get(logs_url, headers=headers, allow_redirects=True)

            if response.status_code == 404:
                return {
                    "status": "success",
                    "message": "Logs not available (may have expired or run is still in progress).",
                    "run_id": run_id,
                    "workflow": run.name,
                    "run_status": run.status,
                    "conclusion": run.conclusion,
                    "jobs": job_info,
                }
            elif response.status_code != 200:
                return {
                    "error": f"Failed to download logs: HTTP {response.status_code}",
                    "run_id": run_id,
                    "jobs": job_info,
                }

            # Extract logs from zip
            filter_job = job_name
            if failed_only and not job_name:
                # If failed_only but no specific job, we'll filter in extraction
                pass

            logs = self._extract_logs_from_zip(
                response.content,
                job_name=filter_job,
                failed_jobs=failed_jobs if failed_only else None,
                tail_lines=tail_lines,
            )

            # Filter to failed jobs if requested
            if failed_only and not job_name:
                logs = {k: v for k, v in logs.items() if any(fj.lower() in k.lower() for fj in failed_jobs)}

            result: Dict[str, Any] = {
                "status": "success",
                "run_id": run_id,
                "workflow": run.name,
                "run_status": run.status,
                "run_conclusion": run.conclusion,
                "run_url": run.html_url,
                "head_sha": run.head_sha[:8],
                "jobs_summary": {
                    "total": len(list(jobs)),
                    "failed": len(failed_jobs),
                },
                "tail_lines": tail_lines,
            }

            if job_name:
                result["filtered_job"] = job_name

            if failed_only:
                result["failed_only"] = True
                result["failed_jobs"] = failed_jobs

            # Add logs (truncate if too large)
            total_log_size = sum(len(v) for v in logs.values())
            if total_log_size > 50000:  # ~50KB limit
                result["warning"] = "Logs truncated due to size. Use job_name filter for specific job logs."
                # Truncate each log
                truncated_logs = {}
                for k, v in logs.items():
                    if len(v) > 10000:
                        truncated_logs[k] = f"... [truncated, showing last portion] ...\n{v[-10000:]}"
                    else:
                        truncated_logs[k] = v
                result["logs"] = truncated_logs
            else:
                result["logs"] = logs

            return result

        except GithubException as e:
            error_msg = e.data.get("message", str(e)) if hasattr(e, "data") else str(e)
            return {"error": f"GitHub API error: {error_msg}"}

        except requests.RequestException as e:
            return {"error": f"Failed to download logs: {str(e)}"}

        except Exception as e:
            logger.exception(f"Error getting CI logs: {e}")
            return {"error": f"Failed to get CI logs: {str(e)}"}
