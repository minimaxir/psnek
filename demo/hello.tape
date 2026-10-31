# Render from the repository root with: python demo/render.py demo/hello.tape
# It records wider and taller than needed, then crops the videos to the text.
Output demo/psnek_hello.mp4
Output demo/psnek_hello.gif

# Pure black background and near-white text for maximum contrast.
Set Theme "Tomorrow Night Bright"
# Ghostty's default font and spacing, so the demo looks like a real terminal.
Set FontFamily "JetBrains Mono"
Set FontSize 16
Set Width 1200
Set Height 400
Set Padding 32
Set LineHeight 1.0
Set LetterSpacing 0
Set Framerate 30
Set TypingSpeed 20ms

Env COLORTERM "truecolor"
Env PSNEK_MODE "live"

# Python 3.14's REPL colors its syntax; VHS records the colors it prints.
Hide
Type "source .venv/bin/activate && alias python='uv run --quiet --isolated --no-dev --python 3.14+gil python' && clear"
Enter
Sleep 500ms
Show

# Each line waits for the REPL's prompt, so no keystroke is echoed ahead of it.
Type "python -q"
Sleep 300ms
Enter
Sleep 800ms
Type "import time, psnek"
Enter
Sleep 800ms
Type "numbers = list(range(1, 121))"
Enter
Sleep 300ms
Type "total = 0"
Enter
Sleep 300ms
Type "for number in psnek.track(numbers):"
Enter
Sleep 300ms
# The REPL indents the loop body itself.
Type "total += number"
Enter
Sleep 300ms
Type "time.sleep(0.03)"
Enter
Sleep 300ms
Enter
Sleep 4800ms
Type "print(total)"
Enter
Sleep 1500ms
