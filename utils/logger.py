import logging
import sys
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager

from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, \
    TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

from configs.config import LOGGING_CONFIG

console = Console()

class EnhancedColoredLogger:
    def __init__(self, name: str, log_config: dict = None):
        self.name = name
        self.config = log_config or LOGGING_CONFIG
        self.console = console
        self.start_time = datetime.now()
        self.log_history = []
        self.error_count = 0
        self.warning_count = 0
        self._suppress_console_output = False  # Flag to control console output
        self._suppress_patterns = self.config.get("suppress_patterns", [])
        self._suppress_during_progress = self.config.get("suppress_during_progress", True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, self.config["level"]))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with rich formatting
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            show_time=True,
            markup=True,
            console=self.console
        )
        console_handler.setLevel(getattr(logging, self.config["console_level"]))
        
        # File handler with rotation
        log_file_path = Path(self.config["log_file"])
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Daily rotation for better log management
        file_handler = TimedRotatingFileHandler(
            self.config["log_file"],
            when="midnight",
            interval=1,
            backupCount=self.config["backup_count"],
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, self.config["file_level"]))
        
        # Formatter for file handler with more details
        file_formatter = logging.Formatter(
            self.config["format"], 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Log system initialization
        self._log_system_info("initialized")
    
    def _log_system_info(self, event: str):
        """Log system information"""
        import platform
        try:
            system_info = {
                "event": event,
                "logger_name": self.name,
                "python_version": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "timestamp": datetime.now().isoformat()
            }
            self.logger.info(f" System {event}: {system_info}")
        except Exception as e:
            self.logger.info(f" Logger {event} (system info unavailable: {e})")
    
    def debug(self, message: str, extra: Optional[Dict] = None):
        """Debug logging with extra context"""
        self._log_with_context("debug", message, extra)
    
    def info(self, message: str, extra: Optional[Dict] = None):
        """Info logging with extra context"""
        self._log_with_context("info", message, extra)
    
    def warning(self, message: str, extra: Optional[Dict] = None):
        """Warning logging with extra context"""
        self.warning_count += 1
        self._log_with_context("warning", message, extra)
    
    def error(self, message: str, extra: Optional[Dict] = None, exc_info: bool = True):
        """error logging with extra context and exception info"""
        self.error_count += 1
        self._log_with_context("error", message, extra, exc_info)
    
    def critical(self, message: str, extra: Optional[Dict] = None, exc_info: bool = True):
        """Critical logging with extra context and exception info"""
        self._log_with_context("critical", message, extra, exc_info)
    
    def suppress_console_output(self, suppress: bool = True):
        """Control whether to suppress console output"""
        self._suppress_console_output = suppress
    
    @contextmanager
    def console_output_suppressed(self):
        """Context manager to temporarily suppress console output"""
        old_suppress = self._suppress_console_output
        self._suppress_console_output = True
        try:
            yield
        finally:
            self._suppress_console_output = old_suppress
    
    def _should_suppress_message(self, message: str, level: str) -> bool:
        """Check if message should be suppressed based on patterns and settings"""
        # Don't suppress errors and critical messages
        if level in ["error", "critical"]:
            return False
        
        # Check if message matches any suppress pattern
        if self._suppress_patterns:
            for pattern in self._suppress_patterns:
                if pattern.lower() in message.lower():
                    return True
        
        return False
    
    def _log_with_context(self, level: str, message: str, extra: Optional[Dict] = None, exc_info: bool = False):
        """Internal method to log with context"""
        # Check if message should be suppressed
        if self._should_suppress_message(message, level):
            return
        
        # Add emoji prefixes for better visual distinction
        emoji_map = {
            "debug": "",
            "info": "",
            "warning": "",
            "error": "",
            "critical": ""
        }
        
        emoji_message = f"{emoji_map.get(level, '')} {message}"
        
        # Store in history
        log_entry = {
            "timestamp": datetime.now(),
            "level": level,
            "message": message,
            "extra": extra or {},
            "logger_name": self.name
        }
        self.log_history.append(log_entry)
        
        # Keep only last 1000 entries to prevent memory issues
        if len(self.log_history) > 1000:
            self.log_history = self.log_history[-1000:]
        
        # Skip console output if suppressed (but still log to file)
        if self._suppress_console_output and level in ["info", "debug"]:
            # Only log to file handlers, skip console handlers
            for handler in self.logger.handlers:
                if not isinstance(handler, RichHandler):  # RichHandler is for console
                    # Prepare log data
                    if extra:
                        extra_data = {
                            "extra_context": extra,
                            "error_count": self.error_count,
                            "warning_count": self.warning_count,
                            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
                        }
                        
                        if exc_info:
                            handler.emit(self.logger.makeRecord(
                                self.name, getattr(logging, level.upper()), '', 0, emoji_message, (), None, func='', extra=extra_data, exc_info=True
                            ))
                        else:
                            handler.emit(self.logger.makeRecord(
                                self.name, getattr(logging, level.upper()), '', 0, emoji_message, (), None, func='', extra=extra_data
                            ))
                    else:
                        handler.emit(self.logger.makeRecord(
                            self.name, getattr(logging, level.upper()), '', 0, emoji_message, (), None, func='', exc_info=exc_info
                        ))
            return
        
        # Normal logging to all handlers
        if extra:
            extra_data = {
                "extra_context": extra,
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
            }
            
            if exc_info:
                getattr(self.logger, level)(emoji_message, extra=extra_data, exc_info=True)
            else:
                getattr(self.logger, level)(emoji_message, extra=extra_data)
        else:
            if exc_info:
                getattr(self.logger, level)(emoji_message, exc_info=True)
            else:
                getattr(self.logger, level)(emoji_message)
    
    def log_operation_start(self, operation: str, details: Optional[Dict] = None):
        """Log operation start with context"""
        details_str = f" ({details})" if details else ""
        self.info(f" Starting operation: {operation}{details_str}", extra={"operation": operation, "stage": "start"})
    
    def log_operation_end(self, operation: str, success: bool = True, details: Optional[Dict] = None):
        """Log operation end with context"""
        status_icon = "✅" if success else ""
        status_text = "completed successfully" if success else "failed"
        details_str = f" ({details})" if details else ""
        self.info(f"{status_icon} Operation {operation} {status_text}{details_str}", extra={"operation": operation, "stage": "end", "success": success})
    
    def log_metrics(self, metrics: Dict[str, Any], context: str = "system"):
        """Log metrics with structured format"""
        metrics_text = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        self.info(f" Metrics [{context}]: {metrics_text}", extra={"metrics": metrics, "context": context})
    
    def log_performance(self, operation: str, duration: float, success: bool = True):
        """Log performance metrics"""
        status_icon = "✅" if success else ""
        self.info(f"{status_icon} Performance - {operation}: {duration:.3f}s", extra={"operation": operation, "duration": duration, "success": success})
    
    def create_log_summary(self) -> Dict[str, Any]:
        """Create comprehensive log summary"""
        summary = {
            "logger_name": self.name,
            "total_logs": len(self.log_history),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "start_time": self.start_time.isoformat(),
            "current_time": datetime.now().isoformat(),
            "recent_errors": [entry for entry in self.log_history[-10:] if entry["level"] in ["error", "critical"]],
            "recent_warnings": [entry for entry in self.log_history[-10:] if entry["level"] == "warning"]
        }
        
        self.info(f" Log summary generated: {summary['total_logs']} entries, {summary['error_count']} errors, {summary['warning_count']} warnings")
        return summary
    
    def print_log_summary(self):
        """Print formatted log summary to console"""
        summary = self.create_log_summary()
        
        # Create rich table
        table = Table(title=f" Log Summary for {self.name}", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Logs", str(summary["total_logs"]))
        table.add_row("Errors", str(summary["error_count"]))
        table.add_row("Warnings", str(summary["warning_count"]))
        table.add_row("Uptime", f"{summary['uptime_seconds']:.1f} seconds")
        
        self.console.print(table)
        
        # Recent errors and warnings
        if summary["recent_errors"]:
            error_panel = Panel(
                "\n".join([f" {entry['timestamp']}: {entry['message']}" for entry in summary["recent_errors"]]),
                title="Recent Errors",
                border_style="red"
            )
            self.console.print(error_panel)
        
        if summary["recent_warnings"]:
            warning_panel = Panel(
                "\n".join([f" {entry['timestamp']}: {entry['message']}" for entry in summary["recent_warnings"]]),
                title="Recent Warnings",
                border_style="yellow"
            )
            self.console.print(warning_panel)


class EnhancedProgressManager:
    """Progress manager with improved logging integration"""
    
    def __init__(self, logger_instance: Optional[EnhancedColoredLogger] = None):
        self.console = console
        self.progress = None
        self.tasks = {}
        self.logger = logger_instance or get_logger(__name__)
        
        # Import the version from progress_utils to avoid duplication
        try:
            from utils.progress_utils import EnhancedProgressManager as ProgressUtilsManager
            self._progress_manager = ProgressUtilsManager(logger_instance)
            self.logger.info(" Using progress manager from progress_utils")
        except ImportError:
            self._progress_manager = None
            self.logger.info(" Using fallback progress manager")
    
    def start_main_progress(self, total: int, description: str = "Processing") -> str:
        """Start main progress bar with styling and logging"""
        if self._progress_manager:
            return self._progress_manager.start_main_progress(total, description)
        
        # Fallback implementation
        self.logger.info(f" Starting progress: {description} (total: {total})")
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
            expand=True,
            transient=False
        )
        self.progress.start()
        task_id = self.progress.add_task(f"[cyan]{description}", total=total)
        self.tasks[task_id] = task_id
        return task_id
    
    def update_task(self, task_id: str, advance: float = 1.0, description: Optional[str] = None):
        """Update task progress with logging"""
        if self._progress_manager:
            self._progress_manager.update_task(task_id, advance, description)
        elif task_id in self.tasks and self.progress:
            self.progress.update(self.tasks[task_id], advance=advance)
            if description:
                self.progress.update(self.tasks[task_id], description=f"[green]{description}")
            
            # Log progress every 10%
            task = self.progress.tasks[self.tasks[task_id]]
            if task.total > 0:
                percentage = (task.completed / task.total) * 100
                if percentage % 10 == 0:
                    self.logger.debug(f" Progress: {percentage:.0f}% complete")
    
    def complete_task(self, task_id: str, success: bool = True, message: Optional[str] = None):
        """Complete task with logging"""
        if self._progress_manager:
            self._progress_manager.complete_task(task_id, success, message)
        elif task_id in self.tasks and self.progress:
            if success:
                self.progress.update(self.tasks[task_id], completed=self.progress.tasks[self.tasks[task_id]].total)
                status_icon = "✅"
                status_text = "completed"
            else:
                self.progress.update(self.tasks[task_id], description=f"[red] Failed")
                status_icon = ""
                status_text = "failed"
            
            if message:
                self.logger.info(f"{status_icon} {message}")
            else:
                self.logger.info(f"{status_icon} Task {status_text}")
    
    def stop_all(self):
        """Stop all progress bars"""
        if self._progress_manager:
            self._progress_manager.stop_all()
        elif self.progress:
            self.progress.stop()
            self.tasks.clear()
            self.logger.info(" All progress tasks stopped")


# Global instances
def get_logger(name: str, config: Optional[dict] = None) -> EnhancedColoredLogger:
    """Get an colored logger instance"""
    return EnhancedColoredLogger(name, config)

def get_progress_manager(logger_instance: Optional[EnhancedColoredLogger] = None) -> EnhancedProgressManager:
    """Get progress manager instance"""
    return EnhancedProgressManager(logger_instance)


# convenience functions
def log_matrix_operation(logger: EnhancedColoredLogger, operation: str, matrix_shape: tuple, timestamp: str):
    """Log matrix operation with details"""
    logger.info(f" Matrix operation: {operation} | Shape: {matrix_shape} | Time: {timestamp}", 
               extra={"operation": operation, "matrix_shape": matrix_shape, "timestamp": timestamp})

def log_attention_weights(logger: EnhancedColoredLogger, weights: dict, top_k: int = 5):
    """Log attention weights summary with formatting"""
    if not weights:
        logger.warning(" No attention weights to log")
        return
    
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:top_k]
    weight_str = ", ".join([f"{k}: {v:.3f}" for k, v in sorted_weights])
    logger.info(f" Top-{top_k} attention weights: {weight_str}", 
               extra={"attention_weights": dict(sorted_weights), "top_k": top_k})

def log_forest_update(logger: EnhancedColoredLogger, tree_id: int, score: float, action: str):
    """Log forest update operation with context"""
    logger.info(f" Forest update: Tree {tree_id} | Score: {score:.3f} | Action: {action}", 
               extra={"tree_id": tree_id, "score": score, "action": action})

def log_system_startup(logger: EnhancedColoredLogger, system_name: str = "As-ECTS"):
    """Log system startup with welcome message"""
    welcome_message = f"""
 {system_name} System Starting Up...
 
 Adaptive Shapelet Learning for Early Classification of Streaming Time Series
 """
    logger.info(welcome_message, extra={"system_name": system_name, "event": "startup"})

def log_system_shutdown(logger: EnhancedColoredLogger, system_name: str = "As-ECTS"):
    """Log system shutdown with summary"""
    summary = logger.create_log_summary()
    
    shutdown_message = f"""
 {system_name} System Shutting Down...
 
 Session Summary:
   Total Logs: {summary['total_logs']}
   Errors: {summary['error_count']}
   Warnings: {summary['warning_count']}
   Uptime: {summary['uptime_seconds']:.1f} seconds
 """
    
    logger.info(shutdown_message, extra={"system_name": system_name, "event": "shutdown", "summary": summary})