# Project: Exam Watch (hackathon, 36 hours, 4 people)

A 360/wide camera watches a test room and flags cheating:
phone out, looking at other desks, long stares. Each person is followed by their box (no face recognition).
Read README.md for how it works.

## Who owns which file (only edit your own file unless the team agrees)
- Vince: camera setup (capture/camera.py)
- Person 2: vision/detector.py (YOLO models, tiles)
- Person 3: rules/brain.py + settings/config.py (cheating rules and numbers)
- Person 4: ui/dashboard.py + demo

## Rules for Claude
- Write simple, beginner-friendly Python. No fancy tricks.
- Explain things simply.
- Use Windows cmd commands, not PowerShell.
- Don't rename functions other files use (detect, update_all, start, set_frame, add_alerts, set_people).
- Run the code after changing it to make sure there are no errors.
