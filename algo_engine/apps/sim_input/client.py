__package__ = 'algo_engine.apps.sim_input'

import abc
import threading
import time
import tkinter
from collections.abc import Callable, Iterable
from tkinter import ttk
from typing import TypedDict, NotRequired, Any

from . import LOGGER

LOGGER.getChild('Client')


class Action(object):
    class Step(TypedDict):
        name: str
        procedure: Callable
        args: NotRequired[Iterable]
        kwargs: NotRequired[dict[str, Any]]
        comments: NotRequired[str]

    def __init__(self, name: str):
        self.name = name
        self.steps: list[Action.Step] = []

    def append(self, name: str, action: Callable[[..., ...], None], args=None, kwargs=None, comments: str = None) -> None:
        step = Action.Step(name=name, procedure=action)

        if args is not None:
            step['args'] = args

        if kwargs is not None:
            step['kwargs'] = kwargs

        if comments is not None:
            step['comments'] = comments

        self.steps.append(step)

    def __call__(self, ignore_error: bool = False) -> None:
        n = len(self.steps)
        for i, step in enumerate(self.steps, start=1):
            procedure = step['procedure']
            args = step.get('args', [])
            kwargs = step.get('kwargs', {})
            comments = step.get('comments', '')
            try:
                procedure(*args, **kwargs)
            except Exception as e:
                LOGGER.error(f'<Action {self.name}> <step {comments}>({i} / {n}) Failed!')

                if not ignore_error:
                    raise e

            LOGGER.debug(f'<Action {self.name}> <step {comments}>({i} / {n}) Completed!')

        LOGGER.info(f'<Action {self.name}> completed!')

    def __len__(self):
        return len(self.steps)


class AutoWorkClient(object, metaclass=abc.ABCMeta):
    def __init__(self, root=None):
        self.root = root if root is not None else tkinter.Tk()
        self.root.title("Structured GUI Client")
        self.root.geometry("600x500")

        # State variables
        self.worker_thread = None
        self.running = False

        # Create button
        self.button = ttk.Button(root, text="Takeover Input", command=self.toggle_auto_work)
        self.button.pack(pady=10)

        # Create table
        self.table = ttk.Treeview(
            root,
            columns=("Action", "Status", "Comments"),
            show="tree headings",
            height=20
        )
        self.table.heading("Action", text="Action")
        self.table.heading("Status", text="Status")
        self.table.heading("Comments", text="Comments")
        self.table.pack(padx=10, pady=10, fill="both", expand=True)

        self.actions: dict[int, list[Action]] = {}

        # self.style = ttk.Style()
        # self.style.theme_use("vista")  # Choose a modern theme like 'clam', 'vista', or 'alt'

    @abc.abstractmethod
    def listen_signal(self) -> int:
        ...

    def register_action(self, action: Action, signal: int):
        if signal in self.actions:
            self.actions[signal].append(action)
        else:
            self.actions[signal] = [action]

    def toggle_auto_work(self):
        """Toggle the daemon thread to start or stop auto work."""
        if not self.running:
            self.running = True
            self.button.config(text="Release Control")
            self.worker_thread = threading.Thread(target=self.auto_work, daemon=True)
            self.worker_thread.start()
        else:
            self.running = False
            self.button.config(text="Takeover Input")

    def auto_work(self):
        """Generate data for the table in a loop."""
        while self.running:
            signal = self.listen_signal()
            actions = self.actions.get(signal, [])

            # Update table for the current signal
            self.update_table(actions)

            for action_idx, action in enumerate(actions):
                # Update action status to "Executing"
                self.update_status(parent_id=action_idx, status="Executing")

                for step_idx, step in enumerate(action.steps):
                    # Update step status to "Executing"
                    self.update_status(parent_id=action_idx, child_id=step_idx, status="Executing")

                    # Execute the step
                    step["procedure"](*step.get("args", []), **step.get("kwargs", {}))

                    # Mark step as "Done"
                    self.update_status(parent_id=action_idx, child_id=step_idx, status="Done")

                # Mark action as "Done" after all steps
                self.update_status(parent_id=action_idx, status="Done")

    def update_table(self, actions: list[Action]):
        """Render the table based on the provided actions."""
        self.table.delete(*self.table.get_children())  # Clear existing rows

        for action_idx, action in enumerate(actions):
            prefix = chr(0x250C)
            # Insert the parent row
            parent_id = self.table.insert(
                "", "end", iid=action_idx, values=(f"{prefix} {action.name}", "Pending", "")
            )

            # Insert child rows for steps
            for step_idx, step in enumerate(action.steps):
                prefix = f'{chr(0x251C) if step_idx < len(action.steps) - 1 else chr(0x2514)}{chr(0x2500)}'
                self.table.insert(
                    parent_id,
                    "end",
                    iid=f"{action_idx}-{step_idx}",
                    text=f"Step {step_idx + 1}",
                    values=(f"{prefix} Step {step_idx + 1}", "Pending", step.get("comments", "")),
                )

            # Expand parent row by default
            self.table.item(parent_id, open=True)

        # Auto-adjust column widths
        self.adjust_column_widths()

    def update_status(self, parent_id: int, child_id: int = None, status: str = "Pending"):
        """Update the status of the specified row."""
        row_id = f"{parent_id}" if child_id is None else f"{parent_id}-{child_id}"
        values = self.table.item(row_id, "values")

        # Select the row if the status is "Executing"
        match status:
            case "Executing":
                self.table.selection_set(row_id)
                self.table.see(row_id)  # Ensure the row is visible

        self.table.item(row_id, values=(values[0], status, values[2]))

    def adjust_column_widths(self):
        """Automatically adjust column widths based on content."""

        self.table.column("#0", width=30, minwidth=30, stretch=False)

        for col in self.table["columns"]:
            max_width = max(
                [len(str(self.table.set(item, col))) for item in self.table.get_children()] + [len(col)]
            )
            self.table.column(col, width=max_width * 10, minwidth=30, stretch=True)  # Scale width for readability


class ExampleClient(AutoWorkClient):
    """An example implementation of the AutoWorkClient."""
    dummy_signal = 0
    is_init = False

    @staticmethod
    def dummy_action(name: str, msg: str) -> None:
        LOGGER.info(f'start working on {name}, {msg}')
        time.sleep(2)
        LOGGER.info('working completed!')

    def listen_signal(self) -> int:
        """Simulate listening for a signal (e.g., return random signal)."""
        import random
        if self.is_init:
            delay = 5 + 5 * random.random()
            time.sleep(delay)
        self.dummy_signal += 1
        self.is_init = True
        return 1 + self.dummy_signal % 2


def main():
    client = ExampleClient()

    # Example usage: Register actions for signal 1
    action1 = Action("Example Action 1")
    action1.append(name='S1A1s1', action=ExampleClient.dummy_action, args=("Signal 1 Action 1", "Step 1"))
    action1.append(name='S1A1s2', action=ExampleClient.dummy_action, args=("Signal 1 Action 1", "Step 2"))

    action2 = Action("Example Action 2")
    action2.append(name='S1A2s1', action=ExampleClient.dummy_action, args=("Signal 1 Action 2", "Step 1"))
    action2.append(name='S1A2s2', action=ExampleClient.dummy_action, args=("Signal 1 Action 2", "Step 2"))

    client.register_action(action1, signal=1)
    client.register_action(action2, signal=1)

    action3 = Action("Example Action 3")
    action3.append(name='S2A1s1', action=ExampleClient.dummy_action, args=("Signal 2 Action 1", "Step 1"))
    action3.append(name='S2A1s2', action=ExampleClient.dummy_action, args=("Signal 2 Action 1", "Step 2"))
    action3.append(name='S2A1s3', action=ExampleClient.dummy_action, args=("Signal 2 Action 1", "Step 3"))

    client.register_action(action3, signal=2)

    client.root.mainloop()


if __name__ == "__main__":
    main()
