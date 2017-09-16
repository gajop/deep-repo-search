# Login to Github

from github3 import login
from github3.exceptions import ForbiddenError
gh = login('gajop', password='mokravoda7')

# Obtain a list of all repositories
# Iterate through all repositories, retrieve the metadata and write it to files.

import time
import pickle
import os
import sys
import traceback
import base64

import shutil

# Get the current iterator so we can interrupt the process at any time
dlAmount = 0
try:
    iterFile = open('repo_iter.pkl', 'rb')
    dlProgress = pickle.load(iterFile)
    etag = dlProgress["etag"]
    dlAmount = dlProgress["dlAmount"]
    print("Continuing from where it was left of (%d downloaded)" % dlAmount)
    reposIter = gh.all_repositories(etag=etag)
except:
    reposIter = gh.all_repositories()

repos = []
reposCrawled = 0

REPOS_PER_FILE = 2000
DOWNLOAD_FOLDER = "dl"
FILES = ["README.md", "README", "readme.md", "readme"]

# Later when you start a new process or go to check for new users you can
# then do

# Use this for proper rate limiting
# gh.ratelimit_remaining

def addRepo(repo):
    global repos, reposCrawled, REPOS_PER_FILE
    meta = {"contributors":[], "forks":[], "stargazers":[], "subscribers":[], "teams":[], "files":{}}
    for FILE in FILES:
        f = repo.directory_contents(FILE)
        if f:
            meta["files"][FILE] = base64.b64decode(f)
    for collaborator in repo.contributors():
        meta["contributors"].append(collaborator.as_dict())
        #print(collaborator, type(collaborator), collaborator.as_dict())
    for fork in repo.forks():
        meta["forks"].append(fork.as_dict())
    for stargazer in repo.stargazers():
        meta["stargazers"].append(stargazer.as_dict())
    for subscriber in repo.subscribers():
        meta["subscribers"].append(subscriber.as_dict())
    for team in repo.teams():
        meta["teams"].append(team.as_dict())
    repos.append({"repo":repo.as_dict(), "meta":meta})
    #print(gh.ratelimit_remaining)
    #sys.exit(0)
    #print(repo.as_dict())
    #for cf in repo.code_frequency():
        #print(cf)
    #for ca in repo.commit_activity():
        #print(ca)
    #for cs in repo.contributor_statistics():
        #print(cs)

    #sys.exit(0)
    reposCrawled += 1
    # serialize repos
    #sys.stdout.write("\rCrawled %d repositories." % reposCrawled)
    sys.stdout.write("\rCrawled %d repositories. Ratelimit remaining: %d" % (reposCrawled, gh.ratelimit_remaining))
    sys.stdout.flush()
    if len(repos) >= REPOS_PER_FILE:
        index = 0
        while True:
            fileName = os.path.join(DOWNLOAD_FOLDER, 'repos' + str(index) + '.pkl')
            if not os.path.isfile(fileName):
                break
            index += 1
        pickle.dump(repos, open(fileName, 'wb'))
        repos = []
        #pickle.load(open(fileName, 'rb'))
        #time.sleep(120)

        #sys.exit(0)

def execute():
    if os.path.exists(DOWNLOAD_FOLDER):
        print("Deleting download directory.")
        shutil.rmtree(DOWNLOAD_FOLDER)

    print("Creating download directory.")
    os.makedirs(DOWNLOAD_FOLDER)
    print("Begin repository crawling")
    print("*" * 30)
    for repo in reposIter:
        while gh.ratelimit_remaining < 100:
            print()
            sys.stdout.write("\rRatelimit remaining: %d" % gh.ratelimit_remaining)
            sys.stdout.flush()
            time.sleep(120)
        try:
            addRepo(repo)
            iterFile = open('repo_iter.pkl', 'wb')
            pickle.dump({"etag":reposIter.etag, "dlAmount":dlAmount+reposCrawled}, iterFile)
        except ForbiddenError as e:
            print("ForbiddenError", e)
            traceback.print_exc()
        except Exception as e:
            print("Unknown exception", e)
            traceback.print_exc()

execute()

#while True:
#    reposIter.refresh(True)
#    for repo in reposIter:
#        print(repo)
#        print(reposIter.etag)
#
#    time.sleep(120)  # Sleep for 2 minutes
