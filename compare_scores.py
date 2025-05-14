import matplotlib.pyplot as plt
import csv

def load_scores(filename):
    scores = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            if row: 
                scores.append(int(row[0]))
    return scores

ai_scores = load_scores("ai_scores.csv")
human_scores = load_scores("human_scores.csv")

ai_gen = list(range(1, len(ai_scores)+1))
human_game = list(range(1, len(human_scores)+1))

plt.figure(figsize=(10, 6))
plt.plot(ai_gen, ai_scores, label="AI Score", marker='o', color='blue')
plt.plot(human_game, human_scores, label="Human Score", marker='x', color='green')
plt.xlabel("Game Number / Generation")
plt.ylabel("Score")
plt.title("AI vs Human Performance in Flappy Bird")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("score_comparison.png")
plt.show()
