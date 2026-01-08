
from typing import Dict, Optional, Any, Callable
from datetime import datetime
import threading
from contextlib import contextmanager

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.columns import Columns

from utils.logger import get_logger

logger = get_logger(__name__)


class ProgressManager:
    """Progress manager with multiple progress bars and rich visualization"""
    
    def __init__(self, config: Optional[Dict] = None):
        from configs.config import PROGRESS_CONFIG
        self.config = config or PROGRESS_CONFIG
        
        # Use a dedicated console for progress to avoid conflicts
        self.console = Console(
            width=self.config.get("console_width", 120),
            stderr=False,  # Use stdout for progress
            quiet=False,
            legacy_windows=False,
            force_terminal=True,
            color_system="auto"
        )
        
        self.progress = None
        self.tasks = {}
        self.task_metadata = {}
        self.start_time = datetime.now()
        self.overall_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "current_stage": "initializing"
        }
        
        # Thread safety locks
        self._progress_lock = threading.Lock()
        self._live_context = None
        self._suppress_logging = False
        
        # Logger instance
        self.logger = logger
        
        # Remove initialization logging to avoid progress bar interference
        # logger.info(" Progress manager initialized")

    def start_main_progress(self, total: int, description: str = "Processing", task_id: str = "main") -> str:
        """Start main progress bar with styling and single-line display"""
        with self._progress_lock:
            # Check if we already have an active progress/Live context
            if self.progress and self._live_context:
                # If we already have an active progress, just add a new task to it
                styled_task_id = self.progress.add_task(f"[cyan]{description}", total=total)
                self.tasks[task_id] = styled_task_id
                self.task_metadata[task_id] = {
                    "description": description,
                    "start_time": datetime.now(),
                    "type": "main",
                    "total": total
                }
                self.overall_stats["total_tasks"] += 1
                self.overall_stats["current_stage"] = description
                
                # Remove task addition logging to avoid progress bar interference
                # if not self._suppress_logging:
                #     logger.info(f" Adding task to existing progress: {description} (total: {total})")
                
                return task_id
            
            # Use Live context manager for single-line display
            if self.config.get("use_rich_live", True):
                self.progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(complete_style="green", finished_style="green"),
                    TaskProgressColumn(),
                    TextColumn(""),
                    TimeElapsedColumn(),
                    console=self.console,
                    expand=False,  # Don't expand to full width
                    refresh_per_second=self.config.get("refresh_per_second", 4),  # Reduced refresh rate
                    auto_refresh=self.config.get("auto_refresh", True),
                    transient=True  # Make transient to avoid permanent lines
                )
                
                # Start Live context for single-line display
                self._live_context = Live(
                    self.progress,
                    console=self.console,
                    refresh_per_second=self.config.get("refresh_per_second", 4),  # Reduced refresh
                    transient=True,  # Always transient for clean display
                    redirect_stdout=self.config.get("redirect_stdout", True),
                    redirect_stderr=self.config.get("redirect_stderr", True),
                    screen=False,  # Don't use alternate screen
                    auto_refresh=True
                )
                self._live_context.start()
            else:
                # Fallback to regular progress
                self.progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(complete_style="green", finished_style="green"),
                    TaskProgressColumn(),
                    TextColumn(""),
                    TimeElapsedColumn(),
                    TextColumn(""),
                    TimeRemainingColumn(),
                    console=self.console,
                    expand=self.config.get("expand", True)
                )
                self.progress.start()
            
            # Add task
            styled_task_id = self.progress.add_task(f"[cyan]{description}", total=total)
            self.tasks[task_id] = styled_task_id
            self.task_metadata[task_id] = {
                "description": description,
                "start_time": datetime.now(),
                "type": "main",
                "total": total
            }
            self.overall_stats["total_tasks"] += 1
            self.overall_stats["current_stage"] = description
            
            # Log with suppression control
            # Remove task starting logging to avoid progress bar interference
            # if not self._suppress_logging:
            #     logger.info(f" Starting main progress: {description} (total: {total})")
            
            return task_id

    def add_subtask(self, parent_task_id: str, total: int, description: str) -> str:
        """Add a subtask under a parent task"""
        if not self.progress:
            logger.warning(" Progress not started, starting main progress first")
            self.start_main_progress(total=100, description="Main Task")
        
        task_id = f"{parent_task_id}_{len([t for t in self.tasks.keys() if t.startswith(parent_task_id)])}"
        styled_task = self.progress.add_task(f"[green] {description}", total=total)
        self.tasks[task_id] = styled_task
        self.task_metadata[task_id] = {
            "description": description,
            "parent": parent_task_id,
            "start_time": datetime.now(),
            "type": "subtask",
            "total": total
        }
        self.overall_stats["total_tasks"] += 1
        logger.info(f" Added subtask: {description} (parent: {parent_task_id})")
        return task_id

    def print_status(self, message: str, style: str = "info"):
        """Print status message without interfering with progress bars"""
        if self.console:
            # Use console print with appropriate styling
            style_map = {
                "info": "cyan",
                "success": "green",
                "warning": "yellow",
                "error": "red"
            }
            color = style_map.get(style, "white")
            self.console.print(f"[{color}]{message}[/{color}]")
        else:
            # Fallback to regular print
            print(message)

    @contextmanager
    def suppress_logging(self):
        """Context manager to temporarily suppress logging during progress updates"""
        self._suppress_logging = True
        
        # Also suppress console output from logger if self.logger:
        if self.logger:
            self.logger.suppress_console_output(True)
        
        try:
            yield
        finally:
            self._suppress_logging = False
            # Restore console output
            if self.logger:
                self.logger.suppress_console_output(False)

    def update_task(self, task_id: str, advance: float = 1.0, description: Optional[str] = None):
        """Update task progress with optional description change"""
        if task_id in self.tasks and self.progress:
            with self._progress_lock:
                self.progress.update(self.tasks[task_id], advance=advance)
                if description:
                    self.progress.update(self.tasks[task_id], description=f"[green]{description}")
                    self.task_metadata[task_id]["description"] = description
                
                # Log progress every 10% or at completion (with suppression control)
                if not self._suppress_logging:
                    metadata = self.task_metadata[task_id]
                    current_progress = self.progress.tasks[self.tasks[task_id]].completed
                    total = metadata["total"]
                    
                    if total > 0:
                        percentage = (current_progress / total) * 100
                        if percentage % 10 == 0 or advance >= total:
                            # Use debug level to reduce interference
                            logger.debug(f" {task_id}: {percentage:.0f}% complete")

    def complete_task(self, task_id: str, success: bool = True, message: Optional[str] = None):
        """Mark a task as complete"""
        if task_id in self.tasks and self.progress:
            with self._progress_lock:
                if success:
                    self.progress.update(self.tasks[task_id], completed=self.task_metadata[task_id]["total"])
                    self.overall_stats["completed_tasks"] += 1
                    status_icon = "✅"
                    status_text = "completed"
                else:
                    self.progress.update(self.tasks[task_id], description=f"[red] {self.task_metadata[task_id]['description']}")
                    self.overall_stats["failed_tasks"] += 1
                    status_icon = ""
                    status_text = "failed"
                
                # Calculate duration
                start_time = self.task_metadata[task_id]["start_time"]
                duration = datetime.now() - start_time
                
                # Only log completion messages, skip console print to avoid interference
                if message:
                    if not self._suppress_logging:
                        logger.info(f"{status_icon} {task_id}: {message} (duration: {duration.total_seconds():.1f}s)")
                else:
                    if not self._suppress_logging:
                        logger.info(f"{status_icon} {task_id}: Task {status_text} (duration: {duration.total_seconds():.1f}s)")

    def stop_all(self):
        """Stop all progress bars and show summary"""
        with self._progress_lock:
            if self.progress:
                # Stop Live context first if it exists
                if self._live_context:
                    self._live_context.stop()
                    self._live_context = None
                
                # Then stop progress
                self.progress.stop()
                
                # Show completion summary
                total_time = datetime.now() - self.start_time
                self._show_completion_summary(total_time)
                
                self.tasks.clear()
                self.task_metadata.clear()
                
                if not self._suppress_logging:
                    logger.info(" All progress tasks stopped")

    def _show_completion_summary(self, total_time):
        """Show completion summary table"""
        table = Table(title="Task Completion Summary", show_header=True, header_style="bold magenta")
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Description", style="green")
        table.add_column("Status", justify="center")
        table.add_column("Duration", justify="right")
        table.add_column("Progress", justify="right")
        
        for task_id, metadata in self.task_metadata.items():
            duration = datetime.now() - metadata["start_time"]
            status = "✅ Completed" if metadata.get("completed", False) else " Failed" if metadata.get("failed", False) else " Running"
            
            if self.progress and task_id in self.tasks:
                task = self.progress.tasks[self.tasks[task_id]]
                progress = f"{task.completed}/{task.total}" if task.total else "N/A"
            else:
                progress = "N/A"
            
            table.add_row(
                str(task_id),
                str(metadata["description"]),
                str(status),
                f"{duration.total_seconds():.1f}s",
                str(progress)
            )
        
        self.console.print(table)
        
        # Overall statistics
        stats_table = Table(title="Overall Statistics", show_header=False)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right", style="green")
        
        stats_table.add_row("Total Tasks", str(self.overall_stats["total_tasks"]))
        stats_table.add_row("Completed Tasks", str(self.overall_stats["completed_tasks"]))
        stats_table.add_row("Failed Tasks", str(self.overall_stats["failed_tasks"]))
        stats_table.add_row("Success Rate", f"{(self.overall_stats['completed_tasks'] / max(self.overall_stats['total_tasks'], 1)) * 100:.1f}%")
        stats_table.add_row("Total Time", f"{total_time.total_seconds():.1f}s")
        
        self.console.print(stats_table)
        
        logger.info(f" Task summary: {self.overall_stats['completed_tasks']}/{self.overall_stats['total_tasks']} completed, "
                   f"success rate: {(self.overall_stats['completed_tasks'] / max(self.overall_stats['total_tasks'], 1)) * 100:.1f}%")

    def create_detailed_dashboard(self, system_stats: Optional[Dict] = None) -> str:
        """Create detailed progress dashboard"""
        logger.info(" Creating detailed progress dashboard")
        
        layout = Layout()
        
        # Header
        header = Panel(
            Text(" As-ECTS System Progress Dashboard", style="bold cyan", justify="center"),
            title="System Status",
            border_style="blue"
        )
        
        # Progress section
        progress_panel = self._create_progress_panel()
        
        # Statistics section
        stats_panel = self._create_stats_panel(system_stats)
        
        # Task details section
        details_panel = self._create_task_details_panel()
        
        # System health section
        health_panel = self._create_health_panel(system_stats)
        
        # Arrange layout
        layout.split_column(
            Layout(header, size=3),
            Layout(progress_panel, size=10),
            Layout(stats_panel, size=8),
            Layout(details_panel, size=12),
            Layout(health_panel, size=6)
        )
        
        # Save dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_file = f"progress_dashboard_{timestamp}.txt"
        
        with open(dashboard_file, 'w') as f:
            with self.console.capture() as capture:
                self.console.print(layout)
            f.write(capture.get())
        
        logger.info(f"✅ Dashboard saved to {dashboard_file}")
        return dashboard_file

    def _create_progress_panel(self) -> Panel:
        """Create progress visualization panel"""
        if not self.progress or not self.tasks:
            return Panel("No active progress", title="Progress", border_style="yellow")
        
        # Get current task progress
        progress_bars = []
        for task_id, styled_task in self.tasks.items():
            if task_id in self.task_metadata:
                metadata = self.task_metadata[task_id]
                task = self.progress.tasks[styled_task]
                
                # Create progress bar text
                percentage = (task.completed / task.total * 100) if task.total > 0 else 0
                bar_length = 20
                filled_length = int(bar_length * percentage // 100)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                
                progress_text = f"{metadata['description']}\n[{bar}] {percentage:.1f}% ({task.completed}/{task.total})"
                progress_bars.append(Text(progress_text, style="green"))
        
        return Panel(
            Columns(progress_bars) if progress_bars else Text("No active tasks"),
            title="Current Progress",
            border_style="green"
        )

    def _create_stats_panel(self, system_stats: Optional[Dict]) -> Panel:
        """Create statistics panel"""
        if not system_stats:
            return Panel("No system statistics available", title="Statistics", border_style="yellow")
        
        stats_text = ""
        for key, value in system_stats.items():
            if key != 'timestamp':
                stats_text += f"{key.replace('_', ' ').title()}: {value}\n"
        
        return Panel(
            Text(stats_text, style="cyan"),
            title="System Statistics",
            border_style="cyan"
        )

    def _create_task_details_panel(self) -> Panel:
        """Create task details panel"""
        if not self.task_metadata:
            return Panel("No task details available", title="Task Details", border_style="yellow")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Task", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Status", justify="center")
        table.add_column("Duration", justify="right")
        
        for task_id, metadata in self.task_metadata.items():
            duration = datetime.now() - metadata["start_time"]
            task_type = metadata["type"]
            
            # Determine status
            if self.progress and task_id in self.tasks:
                task = self.progress.tasks[self.tasks[task_id]]
                if task.finished:
                    status = "✅ Complete"
                else:
                    status = " Running"
            else:
                status = " Unknown"
            
            table.add_row(
                task_id,
                task_type,
                status,
                f"{duration.total_seconds():.1f}s"
            )
        
        return Panel(table, title="Task Details", border_style="magenta")

    def _create_health_panel(self, system_stats: Optional[Dict]) -> Panel:
        """Create system health panel"""
        if not system_stats:
            return Panel("System health: Unknown", title="Health", border_style="yellow")
        
        # Simple health calculation
        health_score = 100
        health_status = "Excellent"
        health_color = "green"
        
        if 'matrix_size' in system_stats and 'total_shapelets' in system_stats:
            utilization = (system_stats['total_shapelets'] / system_stats['matrix_size']) * 100
            if utilization > 80:
                health_score -= 20
                health_status = "Good"
                health_color = "yellow"
            elif utilization > 95:
                health_score -= 40
                health_status = "Needs Attention"
                health_color = "red"
        
        health_text = f"Status: {health_status}\nScore: {health_score}/100"
        
        return Panel(
            Text(health_text, style=health_color),
            title="System Health",
            border_style=health_color
        )

    def print_status(self, message: str, style: str = "info"):
        """Print status message with styling"""
        styles = {
            "info": "blue",
            "success": "green",
            "warning": "yellow",
            "error": "red",
            "highlight": "magenta"
        }
        color = styles.get(style, "white")
        styled_message = f"[{color}]{message}[/{color}]"
        self.console.print(styled_message)
        logger.info(f"{style.upper()}: {message}")

    def monitor_function(self, func: Callable, *args, **kwargs) -> Any:
        """Monitor function execution with progress tracking"""
        func_name = func.__name__
        logger.info(f" Monitoring function: {func_name}")
        
        task_id = self.start_main_progress(total=100, description=f"Executing {func_name}")
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            self.complete_task(task_id, success=True, message=f"{func_name} completed successfully")
            return result
        except Exception as e:
            self.complete_task(task_id, success=False, message=f"{func_name} failed: {str(e)}")
            logger.error(f" Function {func_name} failed: {e}")
            raise


# Global instance
def get_progress_manager() -> ProgressManager:
    """Get progress manager instance"""
    return EnhancedProgressManager()


# Convenience functions for backward compatibility
def create_progress_task(total: int, description: str) -> ProgressManager:
    """Create a new progress task"""
    manager = get_progress_manager()
    manager.start_main_progress(total, description)
    return manager


def update_progress_task(manager: ProgressManager, advance: float = 1.0):
    """Update current progress task"""
    if manager.tasks:
        task_id = list(manager.tasks.keys())[0]  # Use first task
        manager.update_task(task_id, advance)


def complete_progress_task(manager: ProgressManager, success: bool = True):
    """Complete current progress task"""
    if manager.tasks:
        task_id = list(manager.tasks.keys())[0]  # Use first task
        manager.complete_task(task_id, success)
    manager.stop_all()


# Global instance with better error handling
class EnhancedProgressManager(ProgressManager):
    """Progress manager with better error handling and logging"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._initialized = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.progress:
                self.stop_all()
        except Exception as e:
            logger.warning(f" Error stopping progress manager: {e}")


# Global instance with better initialization
_progress_manager_instance = None

def get_progress_manager() -> EnhancedProgressManager:
    """Get progress manager instance with lazy initialization"""
    global _progress_manager_instance
    if _progress_manager_instance is None:
        _progress_manager_instance = EnhancedProgressManager()
    return _progress_manager_instance