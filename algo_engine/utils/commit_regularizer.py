import pathlib
import random
from collections import defaultdict
from datetime import datetime,date

import pygit2

# Path to your local git repository
repo_path = pathlib.Path(__file__).parents[2]

# Open the repository
repo = pygit2.Repository(repo_path)

# Dictionary to store commits by date
commits_by_date = defaultdict(list)

# Collect commits by date
for commit in repo.walk(repo.head.target, pygit2.GIT_SORT_TIME | pygit2.GIT_SORT_REVERSE):
    commit_date = datetime.fromtimestamp(commit.commit_time).date()
    commits_by_date[commit_date].append(commit)

# Iterate over each date and update commits
for commit_date, commits in commits_by_date.items():

    if commit_date < date(2024, 3, 1):
        continue

    # Generate a random timestamp between 3am and 4am
    random_hour = 3
    random_minute = random.randint(0, 59)
    random_second = random.randint(0, 59)
    random_time = datetime(commit_date.year, commit_date.month, commit_date.day, random_hour, random_minute, random_second)

    # Convert to timestamp
    new_commit_time = int(random_time.timestamp())

    # Update commits for this date
    for index, commit in enumerate(commits):
        # Calculate committer and author timestamps and timezones
        committer = pygit2.Signature(commit.committer.name, commit.committer.email, new_commit_time, commit.committer.offset)
        author = pygit2.Signature(commit.author.name, commit.author.email, new_commit_time, commit.author.offset)

        # Amend the commit
        repo.amend_commit(commit.id, None, author, committer, commit.message, None)

        # Print for logging or verification
        print(f"Amended commit {commit.id} to {random_time} (index {index + 1} of {len(commits)} for {commit_date})")

print("Amendment complete.")
