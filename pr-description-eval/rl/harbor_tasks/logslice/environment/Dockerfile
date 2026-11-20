# PR #12 of acme/logslice, checked out and ready for review follow-up.
# Build context: task/environment.  docker build --build-arg CONTRIBUTING=1 -t pr-desc-task:latest task/environment
FROM python:3.12-slim

ARG CONTRIBUTING=0

RUN apt-get update \
 && apt-get install -y --no-install-recommends git jq \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir pytest==8.3.3 setuptools==75.6.0

# The agent works as the PR author.
RUN git config --global user.name "Marta Kowalski" \
 && git config --global user.email "marta.kowalski@acme.dev" \
 && git config --global init.defaultBranch main \
 && git config --global advice.detachedHead false

COPY fakegh/gh.py /usr/local/bin/gh
COPY fakegh/pr-12.json /var/lib/fakegh/pr-12.json
COPY setup_repo.sh /opt/task-setup/setup_repo.sh
COPY repo_src/ /opt/task-setup/repo_src/

RUN chmod 755 /usr/local/bin/gh \
 && touch /var/lib/fakegh/calls.log \
 && CONTRIBUTING=$CONTRIBUTING bash /opt/task-setup/setup_repo.sh /opt/task-setup/repo_src \
 && rm -rf /opt/task-setup \
 && pip install --no-cache-dir --no-build-isolation -e /repo \
 && git -C /repo status --porcelain | (! grep .)

ENV GH_TOKEN=ghp_fake_sandbox_token GH_REPO=acme/logslice
WORKDIR /repo
CMD ["bash"]
