# Saves a few small frames from the classroom clip for a quick look (gitignored folder).
import sys, json, cv2
times = json.load(open("perf/clip/classroom_times.json"))
want = [float(x) for x in sys.argv[1:]]
cap = cv2.VideoCapture("perf/clip/classroom.avi")
i = 0
targets = {}
for w in want:
    targets[min(range(len(times)), key=lambda k: abs(times[k] - w))] = w
while True:
    ok, f = cap.read()
    if not ok:
        break
    if i in targets:
        cv2.imwrite("perf/frames/peek_%03d.jpg" % int(targets[i]), cv2.resize(f, (960, 540)))
    i += 1
print("frames", i)
