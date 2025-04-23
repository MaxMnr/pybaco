import time
from typing import Dict
import multiprocessing as mp
import threading
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

class ProgressTracker:
    """
    A class to track progress across multiple processes with Rich library 
    (using tqdm caused to much bugs and was messy).
    """
    def __init__(self, tasks: Dict[str, int], update_interval: float = 0.5):
        self.tasks = tasks
        self.update_interval = update_interval
        self.counters = {}
        self.task_ids = {}
        self.running = False
        self.display_thread = None
        
        # Create shared counters
        for task_name, _ in tasks.items():
            self.counters[task_name] = mp.Value('i', 0)
    
    def increment(self, task_name: str, amount: int = 1):
        # Increment the counter for a specific task.
        with self.counters[task_name].get_lock():
            self.counters[task_name].value += amount
    
    def get_progress(self):
        # Get the current progress for all tasks.
        return {task: counter.value for task, counter in self.counters.items()}
    
    def _display_progress(self):
        # Create a Rich Progress instance
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("[bold green]Step {task.completed} of {task.total}")  # Add custom step display
        ) as progress:
            # Create tasks for each item
            for task_name, total in self.tasks.items():
                self.task_ids[task_name] = progress.add_task(f"[cyan]{task_name}", total=total)
            
            # Update progress regularly
            while self.running:
                for task_name, task_id in self.task_ids.items():
                    current = self.counters[task_name].value
                    progress.update(task_id, completed=current)
                
                time.sleep(self.update_interval)
    
    def start(self):
        """Start the progress display thread."""
        self.running = True
        # Use a regular thread since Rich doesn't work well with multiprocessing
        self.display_thread = threading.Thread(target=self._display_progress)
        self.display_thread.daemon = True
        self.display_thread.start()
    
    def stop(self):
        """Stop the progress display thread and print final status."""
        self.running = False
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join()
