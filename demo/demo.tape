# Render from the repository root with: python demo/render.py demo/demo.tape
# It records wider and taller than needed, then crops the videos to the text.
Output demo/psnek.mp4
Output demo/psnek.gif

# Pure black background and near-white text for maximum contrast.
Set Theme "Tomorrow Night Bright"
Set FontFamily "JetBrains Mono"
Set FontSize 16
Set Width 1200
Set Height 400
Set Padding 32
Set LineHeight 1.0
Set LetterSpacing 0
Set Framerate 30
Set TypingSpeed 40ms

Env COLORTERM "truecolor"
Env PSNEK_MODE "live"

Hide
Type "source .venv/bin/activate && clear"
Enter
Sleep 500ms
Show

Type "python demo/demo.py"
Sleep 300ms
Enter
Sleep 9000ms
