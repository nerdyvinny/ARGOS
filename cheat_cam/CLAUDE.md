# Project: Exam Watch (hackathon, 36 hours, 4 people)

A 360/wide camera watches a test room and flags cheating:
phone out, looking at other desks, long stares. Desk = ID (no face recognition).
Read README.md for how it works.

## Who owns which file (only edit your own file unless the team agrees)
- Vince: camera setup + setup_desks.py
- Person 2: detector.py (Hugging Face models & Datasets, tiles)
- Person 3: brain.py + config.py (cheating rules and numbers)
- Person 4: dashboard.py + demo

## Rules for Claude
- Write simple, beginner-friendly Python. No fancy tricks.
- Explain things simply.
- Use Windows cmd commands, not PowerShell.
- Don't rename functions other files use (detect, update_all, load_desks, start, set_frame, add_alerts, set_desks).
- Run the code after changing it to make sure there are no errors.
