"""
零依赖富文本日志系统
标准模式 + 柔和配色 + 实心方块进度条
"""

import sys
import time
import threading
import re
from datetime import datetime
from typing import Optional


class ColorScheme:
    """ANSI转义码颜色方案 - 柔和配色"""

    RESET = "\033[0m"
    BOLD = "\033[1m"

    SUCCESS = "\033[92m"
    WARNING = "\033[38;5;214m"
    ERROR = "\033[38;5;211m"
    INFO = "\033[38;5;153m"
    PROGRESS = "\033[38;5;141m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    @staticmethod
    def paint(text: str, color: str, bold: bool = False) -> str:
        prefix = ColorScheme.BOLD if bold else ""
        return f"{prefix}{color}{text}{ColorScheme.RESET}"

    _ANSI_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    @staticmethod
    def strip_color(text: str) -> str:
        return ColorScheme._ANSI_PATTERN.sub('', text)


def get_display_width(text: str) -> int:
    """计算字符串的实际显示宽度"""
    width = 0
    for char in text:
        code = ord(char)
        if code <= 0x7F:
            width += 1
        elif 0x4E00 <= code <= 0x9FFF:
            width += 2
        elif 0x3400 <= code <= 0x4DBF:
            width += 2
        elif 0x3000 <= code <= 0x303F:
            width += 2
        elif 0x3040 <= code <= 0x309F:
            width += 2
        elif 0x30A0 <= code <= 0x30FF:
            width += 2
        elif 0xAC00 <= code <= 0xD7AF:
            width += 2
        elif 0x1F300 <= code <= 0x1F9FF:
            width += 2
        elif 0x2600 <= code <= 0x26FF:
            width += 2
        elif 0x2700 <= code <= 0x27BF:
            width += 2
        elif 0xFE00 <= code <= 0xFE0F:
            width += 0
        elif 0xFF00 <= code <= 0xFFEF:
            width += 2
        else:
            width += 1
    return width


_active_progress_bar = None
_progress_bar_lock = threading.Lock()


class ProgressBar:
    """实时进度条"""

    def __init__(self, total: int, description: str = "处理中", width: int = 20):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()
        self.lock = threading.Lock()
        self._last_line_length = 0
        self._last_line = ""

    def clear_line(self):
        if self._last_line_length > 0:
            sys.stdout.write("\r" + " " * self._last_line_length + "\r")
            sys.stdout.flush()

    def restore_line(self):
        if self._last_line:
            sys.stdout.write(self._last_line)
            sys.stdout.flush()

    def update(self, n: int = 1):
        with self.lock:
            self.current += n
            if self.current > self.total:
                self.current = self.total
            self._render()

    def _render(self):
        percent = (self.current / self.total) * 100 if self.total > 0 else 0
        filled = int(self.width * self.current / self.total) if self.total > 0 else 0

        bar = '█' * filled + '░' * (self.width - filled)

        elapsed = time.time() - self.start_time
        if self.current > 0 and elapsed > 0:
            speed = self.current / elapsed
            eta = (self.total - self.current) / speed if speed > 0 else 0
        else:
            eta = 0

        percent_str = ColorScheme.paint(f"{percent:.0f}%", ColorScheme.PROGRESS, bold=True)
        line = (
            f"\r🔄 {self.description}: "
            f"{ColorScheme.paint(bar, ColorScheme.PROGRESS)} "
            f"{percent_str} "
            f"({self.current}/{self.total}) | "
            f"用时: {elapsed:.1f}s"
        )

        if self.current < self.total and eta > 0:
            line += f" | 预计: {eta:.1f}s"

        line_stripped = ColorScheme.strip_color(line)
        current_width = get_display_width(line_stripped)

        if current_width < self._last_line_length:
            line += " " * (self._last_line_length - current_width)

        self._last_line_length = current_width

        sys.stdout.write(line)
        sys.stdout.flush()

        self._last_line = line

        if self.current >= self.total:
            print()

    def __enter__(self):
        global _active_progress_bar
        with _progress_bar_lock:
            _active_progress_bar = self
        return self

    def __exit__(self, *args):
        global _active_progress_bar
        if self.current < self.total:
            self.current = self.total
            self._render()

        with _progress_bar_lock:
            _active_progress_bar = None


class ThreadSafeLogger:
    """线程安全的日志系统"""

    _THREAD_PATTERN = re.compile(r'(\d+)')

    def __init__(self):
        self.lock = threading.Lock()
        self._enable_color = True
        self._check_terminal_support()

    def _check_terminal_support(self):
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception:
                self._enable_color = False

    def _format_message(self, level: str, message: str, emoji: str, color: str) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        thread = threading.current_thread().name
        if thread == "MainThread":
            thread_name = "[Main]".ljust(8)
        else:
            match = self._THREAD_PATTERN.search(thread)
            if match:
                thread_name = f"[T-{match.group(1)}]".ljust(8)
            else:
                thread_name = "[Work]".ljust(8)

        line = f"[{timestamp}]{thread_name} {emoji} {message}"

        if self._enable_color:
            return ColorScheme.paint(line, color)
        return line

    def _print_with_progress_handling(self, message: str):
        global _active_progress_bar

        with self.lock:
            if _active_progress_bar:
                with _progress_bar_lock:
                    _active_progress_bar.clear_line()

            print(message, flush=True)

            if _active_progress_bar:
                with _progress_bar_lock:
                    _active_progress_bar.restore_line()

    def info(self, message: str):
        line = self._format_message("INFO", message, "ℹ️", ColorScheme.INFO)
        self._print_with_progress_handling(line)

    def success(self, message: str):
        line = self._format_message("SUCCESS", message, "✅", ColorScheme.SUCCESS)
        self._print_with_progress_handling(line)

    def warning(self, message: str):
        line = self._format_message("WARNING", message, "⚠️", ColorScheme.WARNING)
        self._print_with_progress_handling(line)

    def error(self, message: str):
        line = self._format_message("ERROR", message, "❌", ColorScheme.ERROR)
        self._print_with_progress_handling(line)

    def progress_bar(self, total: int, description: str = "处理中") -> ProgressBar:
        return ProgressBar(total, description)

    def separator(self, char: str = "=", length: int = 60):
        with self.lock:
            print(ColorScheme.paint(char * length, ColorScheme.GRAY), flush=True)

    def header(self, title: str, width: int = 60):
        with self.lock:
            title_with_spaces = f"  {title}  "
            title_width = get_display_width(title_with_spaces)

            content_width = width - 2

            total_padding = content_width - title_width
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding

            top = "╔" + "═" * content_width + "╗"
            middle = "║" + " " * left_padding + title_with_spaces + " " * right_padding + "║"
            bottom = "╚" + "═" * content_width + "╝"

            print(ColorScheme.paint(top, ColorScheme.PROGRESS), flush=True)
            print(ColorScheme.paint(middle, ColorScheme.PROGRESS, bold=True), flush=True)
            print(ColorScheme.paint(bottom, ColorScheme.PROGRESS), flush=True)

    def summary(self, title: str, items: dict, width: int = 60):
        with self.lock:
            title_line = f"✨ {title}"
            print(f"\n{ColorScheme.paint(title_line, ColorScheme.SUCCESS, bold=True)}", flush=True)

            for key, value in items.items():
                line = f"   {key}: {ColorScheme.paint(str(value), ColorScheme.WHITE, bold=True)}"
                print(line, flush=True)

            print()


logger = ThreadSafeLogger()

__all__ = ['logger', 'ColorScheme', 'ProgressBar']
