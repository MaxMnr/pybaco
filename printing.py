from rich import print as rprint
from rich.panel import Panel
from rich.console import Console
from rich.align import Align

"""
Just a bunch of functions to print prettier stuff
"""

console = Console()

def print_title(text, title="Title Goes Here!", style="gray", border="bold green"):
    centered_text = Align.center(f"[bold {style}]{text}[/bold {style}]")
    panel = Panel.fit(centered_text, title=title, border_style=border)
    console.print(panel, justify="center") 

def print_info(text):
    rprint(f"[bold cyan][INFO][/bold cyan] [white]{text}[/white]")

def print_success(text):
    rprint(f"[bold green][SUCCESS][/bold green] {text}")

def print_warning(text):
    rprint(f"[bold yellow][WARNING][/bold yellow] {text}")

def print_error(text):
    rprint(f"[bold red][ERROR][/bold red] {text}")

def print_remark(text):
    rprint(f"[bold purple]{text}[/bold purple]")

def print_divider(char="=", length=100, color="gray"):
    rprint(f"[{color}]{char * length}[/{color}]")
