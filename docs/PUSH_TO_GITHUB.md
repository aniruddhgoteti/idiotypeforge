# Pushing IdiotypeForge to your personal GitHub

Three options. Pick the one that matches what you have installed.

---

## Option A — `gh` CLI (fastest, recommended)

If you have the [GitHub CLI](https://cli.github.com) installed and authenticated:

```bash
cd ~/workspace/idiotypeforge

# First time only — make the initial commit
git add -A
git commit -m "IdiotypeForge: personalized lymphoma therapy design with Gemma 4"

# Create the repo on your account and push in one shot
gh repo create idiotypeforge \
    --public \
    --source=. \
    --remote=origin \
    --description "From a biopsy to a personalized lymphoma therapy design — in hours, not months. Gemma 4 + AlphaFold + RFdiffusion." \
    --push
```

That's it. The repo is now at `https://github.com/<your-username>/idiotypeforge`.

To check it's authenticated:

```bash
gh auth status
```

If not, `gh auth login` walks you through it.

---

## Option B — Web UI + git push

1. Go to <https://github.com/new>
2. Repository name: `idiotypeforge`
3. Visibility: **Public**
4. **Do NOT** check "Add a README" / "Add .gitignore" / "Choose a license" — we already have those
5. Click **Create repository**

Then in your terminal:

```bash
cd ~/workspace/idiotypeforge

# First time only — make the initial commit
git add -A
git commit -m "IdiotypeForge: personalized lymphoma therapy design with Gemma 4"

# Hook up the remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/idiotypeforge.git

# Push
git branch -M main
git push -u origin main
```

If GitHub asks for credentials and you use 2FA, you need a Personal Access Token (Settings → Developer settings → Personal access tokens → "Tokens (classic)" → Generate new token, scope `repo`). Use the token as the password when prompted.

---

## Option C — SSH (if you have SSH keys set up with GitHub)

```bash
cd ~/workspace/idiotypeforge

git add -A
git commit -m "IdiotypeForge: personalized lymphoma therapy design with Gemma 4"

git remote add origin git@github.com:YOUR_USERNAME/idiotypeforge.git
git branch -M main
git push -u origin main
```

To check SSH is set up: `ssh -T git@github.com` should say "Hi YOUR_USERNAME!".

---

## After pushing

The repo is live. You may want to:

1. **Add a description and topics on GitHub.** From the repo page → ⚙️ next to "About" → add the description (suggested: *"Personalized lymphoma therapy design with Gemma 4, AlphaFold, RFdiffusion"*) and topics: `lymphoma`, `antibody-design`, `gemma`, `alphafold`, `rfdiffusion`, `kaggle-hackathon`, `oncology`, `personalized-medicine`.

2. **Update README placeholders.** Run this to swap your username throughout:
   ```bash
   YOUR=YOUR_USERNAME
   sed -i '' "s/YOUR_USERNAME/$YOUR/g" README.md docs/PUSH_TO_GITHUB.md docs/writeup.md
   git commit -am "docs: set GitHub username"
   git push
   ```
   (On Linux drop the `''` after `-i`.)

3. **Add the Kaggle Writeup link later.** Once you submit on Kaggle (Day 19), add the writeup URL to the README badges.

4. **Add the live demo URL later.** Day 14 deploys the Hugging Face Space; paste its URL into the README "Quickstart" section.

---

## Sanity check before you push

A 30-second check that nothing private leaks:

```bash
cd ~/workspace/idiotypeforge

# 1. No private endpoints in production code (app/ and scripts/ only —
#    this doc and the test file themselves contain the token by design).
grep -r "alphafold-scheduler" app/ scripts/ 2>/dev/null && echo "✗ FAIL — found private endpoint" || echo "✓ no private endpoints"

# 2. No env files / keys
ls -la | grep -E "\.(env|key|pem)$" && echo "✗ FAIL — secret-like files present" || echo "✓ no secret files"

# 3. .gitignore is in place
test -f .gitignore && echo "✓ .gitignore present" || echo "✗ FAIL — .gitignore missing"
```

All three should print a `✓`. If any prints `✗`, fix before pushing.
