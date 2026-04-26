## Dog Adoption Project
## Belmont University
## Jayden Cruz
## DSC-4900-01: Data Science Project/Portfolio (Spring 2026)

## figure.py
# creates figure of dog bucket distribution
# Make sure main.py has been run (model is saved)

import pandas as pd
import matplotlib.pyplot as plt

# load data and filter to dogs only
train = pd.read_csv("/Users/jaydencruz/PycharmProjects/DogProject/data/train.csv")
dogs = train[train["Type"] == 1].copy()

# count how many dogs fall into each adoption speed category
speed_counts = dogs["AdoptionSpeed"].value_counts().sort_index()

# labels for each bar
labels = ["Not Adopted\n(Speed 4)", "3 Months\n(Speed 3)", "1 Month\n(Speed 2)", "1 Week\n(Speed 1)", "Same Day\n(Speed 0)"]
values = [speed_counts[4], speed_counts[3], speed_counts[2], speed_counts[1], speed_counts[0]]
colors = ["#999999", "#c4622d", "#d4a843", "#5b8db8", "#4a7c59"]

# create the bar chart
fig, ax = plt.subplots(figsize=(7, 4.5))

bars = ax.bar(labels, values, color=colors, width=0.55)

# add count labels on top of each bar
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            str(val), ha="center", fontsize=13, fontweight="bold")

# titles and labels
ax.set_title("Dog Adoption Speed Distribution", fontsize=16, fontweight="bold", pad=12)
ax.set_xlabel("Adoption Category (higher score = adopted faster)", fontsize=12)
ax.set_ylabel("Number of Dogs", fontsize=12)

# remove top and right borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("/Users/jaydencruz/PycharmProjects/DogProject/outputs/poster_score_distribution.png", dpi=200, bbox_inches="tight")


# percentage distribution instead of counts
percent = dogs["AdoptionSpeed"].value_counts(normalize=True).sort_index() * 100

plt.figure()
plt.bar(percent.index, percent.values)
plt.title("Adoption Speed (%)")
plt.xlabel("Speed")
plt.ylabel("Percent")
plt.savefig("/Users/jaydencruz/PycharmProjects/DogProject/outputs/adoption_speed_percent.png")
