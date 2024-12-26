__package__ = 'algo_engine.apps.sim_input'

import abc
import time
import tkinter
from collections.abc import Callable, Iterable
from threading import Thread
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

    def append(self, name: str, action: Callable, args=None, kwargs=None, comments: str = None) -> None:
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
    def __init__(self, root=None, **kwargs):
        self.root = root if root is not None else tkinter.Tk()
        self.root.title(kwargs.get('title', "Structured GUI Client"))

        self.actions: dict[int, list[Action]] = {}
        self.layout = {}

        # State variables
        self.worker_thread = None
        self.running = False
        self.recording = False

        self.render_layout()

    def render_layout(self):
        # No fixed geometry, letting it resize
        # self.root.geometry("600x500")

        # Grid(0, 0): Create button
        button_takeover = self.layout['button_takeover'] = ttk.Button(self.root, text="Takeover Input", command=self.toggle_auto_work)
        button_takeover.grid(row=0, column=0, pady=10, sticky="ew")

        # Grid(0, 1): Mock button
        button_record = self.layout['button_record'] = ttk.Button(self.root, text="Record Action", command=self.record_action)
        button_record.grid(row=0, column=1, pady=10, sticky="ew")

        # Grid(1, 0): Mock button
        button_mock = self.layout['button_mock'] = ttk.Button(self.root, text="Mock Action", command=self.toggle_mock)
        button_mock.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # Grid(1, 1): Dropdown for action selection
        action_selected = self.layout['action_selected'] = tkinter.StringVar()
        action_dropdown = self.layout['action_dropdown'] = ttk.Combobox(self.root, textvariable=action_selected, state="readonly")
        action_dropdown.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # Grid(2, 0): Create table
        action_table = self.layout['action_table'] = ttk.Treeview(self.root, columns=("Action", "Status", "Comments"), show="tree headings")
        action_table.heading("Action", text="Action")
        action_table.heading("Status", text="Status")
        action_table.heading("Comments", text="Comments")
        action_table.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Make the window resize properly by configuring grid weights
        self.root.grid_columnconfigure(0, weight=1, minsize=200)  # Adjust minsize for column 0
        self.root.grid_columnconfigure(1, weight=1, minsize=200)  # Adjust minsize for column 1
        self.root.grid_rowconfigure(2, weight=1)  # Make row 2 expand with the window

    @abc.abstractmethod
    def listen_signal(self) -> int:
        ...

    def register_action(self, action: Action, signal: int):
        self.actions.setdefault(signal, []).append(action)

        # Update the dropdown with all registered actions.
        all_actions = [f"Signal {signal} - {action.name}" for signal, actions in self.actions.items() for action in actions]
        action_dropdown = self.layout['action_dropdown']
        action_dropdown["values"] = all_actions

    def mock_action(self, action: Action, timeout: float = 3.):
        """Mock the execution of an action on a transparent test ground."""
        self.root.iconify()

        # Create a borderless, transparent test ground window
        testground = tkinter.Toplevel(self.root)
        # testground.overrideredirect(True)  # Remove all window borders and decorations

        # Get screen dimensions and set the testground window to cover the entire screen
        # screen_width = self.root.winfo_screenwidth()
        # screen_height = self.root.winfo_screenheight()
        # testground.geometry(f"{screen_width}x{screen_height}+0+0")

        # Make the window semi-transparent and ensure it stays on top
        testground.attributes("-alpha", 0.5)  # Set transparency to 50%
        testground.attributes("-fullscreen", True)
        testground.attributes("-topmost", True)  # Ensure it stays above other windows

        # Force the window manager to render the window completely
        testground.lift()  # Bring it to the front
        testground.focus_force()  # Force focus to this window

        # Add a label to display input history
        input_history = tkinter.Listbox(testground, font=("Courier", 14))
        input_history.pack(fill="both", expand=True, padx=20, pady=20)
        input_history.insert("active", f"Mocking {action.name}")
        testground.update_idletasks()

        # Simulate action execution
        for step in action.steps:
            # Display the step name in the input history
            input_history.insert("end", f"Executing Step: {step['name']}")
            input_history.insert("end", f"Comments: {step.get('comments', 'No comments')}")
            input_history.insert("end", "-" * 50)
            input_history.see("end")
            testground.update_idletasks()

            # Simulate mouse and keyboard inputs
            procedure = step['procedure']
            args = step.get('args', [])
            kwargs = step.get('kwargs', {})
            try:
                procedure(*args, **kwargs)
            except Exception as e:
                input_history.insert("end", f"Error: {str(e)}")
                input_history.see("end")

            # Simulate delay between steps

        # Wait 3 seconds, then close the test ground and restore the client window
        input_history.insert("end", "-" * 50)
        input_history.insert("active", f"Mocking {action.name} Completed! Exiting in {timeout} seconds...")

        time.sleep(timeout)
        testground.destroy()
        self.root.deiconify()

    def toggle_auto_work(self):
        """Toggle the daemon thread to start or stop auto work."""
        if not self.running:
            self.takeover_control()
        else:
            self.release_control()

    def takeover_control(self):
        if self.running:
            LOGGER.info('Autoworker already running!')
            return

        self.running = True
        self.layout['button_takeover'].config(text="Release Control")
        self.worker_thread = Thread(target=self.auto_work, daemon=True)
        self.worker_thread.start()

    def release_control(self):
        if not self.running:
            LOGGER.info('Autoworker already stopped!')
            return

        self.running = False
        self.layout['button_takeover'].config(text="Takeover Input")

    def toggle_mock(self):
        """Mock the action selected in the dropdown."""
        selected_name = self.layout['action_selected'].get()
        if not selected_name:
            LOGGER.warning("No action selected to mock!")
            return

        selected_action = None
        for signal, actions in self.actions.items():
            for action in actions:
                if f"Signal {signal} - {action.name}" == selected_name:
                    selected_action = action
                    break

        if selected_action is None:
            LOGGER.warning(f"Action '{selected_name}' not found!")
            return

        self.mock_action(action=selected_action)

    def auto_work(self):
        """Generate data for the table in a loop."""
        while self.running:
            signal = self.listen_signal()
            actions = self.actions.get(signal, [])

            # Update table for the current signal
            self.update_table(actions)

            for action_idx, action in enumerate(actions):
                if not self.running:
                    break

                # Update action status to "Executing"
                self.update_status(parent_id=action_idx, status="Executing")
                action_done = False

                for step_idx, step in enumerate(action.steps):
                    if not self.running:
                        break

                    # Update step status to "Executing"
                    self.update_status(parent_id=action_idx, child_id=step_idx, status="Executing")

                    # Execute the step
                    step["procedure"](*step.get("args", []), **step.get("kwargs", {}))

                    # Mark step as "Done"
                    self.update_status(parent_id=action_idx, child_id=step_idx, status="Done")
                else:
                    # Mark action as "Done" after all steps
                    self.update_status(parent_id=action_idx, status="Done")
                    action_done = True

                if not action_done:
                    self.update_status(parent_id=action_idx, status="Stopped")

    def update_table(self, actions: list[Action]):
        """Render the table based on the provided actions."""
        action_table = self.layout['action_table']
        action_table.delete(*action_table.get_children())  # Clear existing rows

        for action_idx, action in enumerate(actions):
            prefix = chr(0x250C)
            # Insert the parent row
            parent_id = action_table.insert(
                "", "end", iid=action_idx, values=(f"{prefix} {action.name}", "Pending", "")
            )

            # Insert child rows for steps
            for step_idx, step in enumerate(action.steps):
                prefix = f'{chr(0x251C) if step_idx < len(action.steps) - 1 else chr(0x2514)}{chr(0x2500)}'
                action_table.insert(
                    parent_id,
                    "end",
                    iid=f"{action_idx}-{step_idx}",
                    text=f"Step {step_idx + 1}",
                    values=(f"{prefix} Step {step_idx + 1}", "Pending", step.get("comments", "")),
                )

            # Expand parent row by default
            action_table.item(parent_id, open=True)

        # Auto-adjust column widths
        self.adjust_column_widths()

    def update_status(self, parent_id: int, child_id: int = None, status: str = "Pending"):
        """Update the status of the specified row."""
        row_id = f"{parent_id}" if child_id is None else f"{parent_id}-{child_id}"
        action_table = self.layout['action_table']
        values = action_table.item(row_id, "values")

        # Select the row if the status is "Executing"
        match status:
            case "Executing":
                action_table.selection_set(row_id)
                action_table.see(row_id)  # Ensure the row is visible

        action_table.item(row_id, values=(values[0], status, values[2]))

    def adjust_column_widths(self):
        """Automatically adjust column widths based on content."""
        action_table = self.layout['action_table']
        action_table.column("#0", width=30, minwidth=30, stretch=False)

        for col in action_table["columns"]:
            max_width = max(
                [len(str(action_table.set(item, col))) for item in action_table.get_children()] + [len(col)]
            )
            action_table.column(col, width=max_width * 10, minwidth=30, stretch=True)  # Scale width for readability

    def record_action(self):
        """Capture and log mouse and keyboard events in the test ground."""
        self.root.iconify()

        # Create a semi-transparent test ground window
        testground = tkinter.Toplevel(self.root)
        testground.attributes("-alpha", 0.5)  # Set transparency
        testground.attributes("-fullscreen", True)
        testground.attributes("-topmost", True)
        testground.lift()
        testground.focus_force()

        # Listbox to display recorded events
        event_log = tkinter.Listbox(testground, font=("Courier", 14), selectmode="none", state=tkinter.DISABLED)
        # event_log.bindtags((event_log, self.root, "all"))
        event_log.pack(fill="both", expand=True, padx=0, pady=0)

        def log_event(event_type, event):
            """Log the event details to the console and the event log."""
            if event_type == "Key":
                event_details = f"{event_type}: {event.keysym} (char: {event.char})"
            else:
                event_details = f"{event_type}: Button-{event.num} @ ({event.x}, {event.y})"
            LOGGER.info(event_details)
            event_log.configure(state=tkinter.NORMAL)
            event_log.insert("end", event_details)
            event_log.see("end")
            event_log.configure(state=tkinter.DISABLED)

        # Bind events for mouse clicks, double-clicks, and keyboard inputs
        testground.bind("<Button-1>", lambda e: log_event("Click", e))
        testground.bind("<Double-1>", lambda e: log_event("Double Click", e))
        testground.bind("<Button-3>", lambda e: log_event("Right Click", e))
        testground.bind("<Key>", lambda e: log_event("Key", e))

        # Exit recording when the ESC key is pressed
        def exit_record(event):
            if event.keysym == "Escape":
                LOGGER.info("Exiting recording mode.")
                testground.destroy()
                self.root.deiconify()

        testground.bind("<Escape>", exit_record)


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
    t = Thread(target=main, daemon=False)
    t.start()
