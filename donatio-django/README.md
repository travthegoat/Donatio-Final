# Donatio-Backend

A Transparent Donation Platform for Myanmar 🇲🇲

---

## 📁 Project Structure Suggestion

To keep things organized, create a parent folder and clone all Donatio-related repos inside:

```bash
mkdir Donatio && cd Donatio

# Clone backend repo
git clone https://github.com/Technortal-Assemble/Donatio-Backend.git

# You can also clone other repos here later
# git clone https://github.com/Technortal-Assemble/Donatio-Frontend.git
# git clone https://github.com/Technortal-Assemble/Donatio-AI.git
```

---

## ⚙️ Tech Stack

- **Backend**: Django
- **Dependency Manager**: [uv](https://github.com/astral-sh/uv)
- **Formatter & Linter**: [ruff](https://docs.astral.sh/ruff/)

---

## 🚀 Getting Started

### 1. Clone the backend repo

```bash
cd Donatio
git clone https://github.com/Technortal-Assemble/Donatio-Backend.git
cd Donatio-Backend
```

### 2. Install `uv`

#### On macOS / Linux

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

#### On Windows (PowerShell)

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Make sure `uv` is added to your system `PATH`.

---

## 🔐 Environment Setup

Create your own `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` to match your local database and secret key setup.

---

## 📦 Dependency Management with `uv`

### Sync all dependencies (from `pyproject.toml`)

```bash
uv sync
```

### Add new packages

```bash
uv add <package-name>
# Example:
uv add django-environ
```

`uv` will update your `pyproject.toml` and keep everything locked.

---

## 🛠️ Run the Dev Server

```bash
uv run manage.py createsuperuser
uv run manage.py makemigrations
uv run manage.py migrate
uv run manage.py runserver
```

You can use `uv run` for **any** Django or Python command.

---

## 🧹 Code Quality with `ruff`

Keep your code clean and consistent:

```bash
# Format code
uv run ruff format .

# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check . --fix
```

---

## ✅ Summary

| Task                  | Command                            |
|-----------------------|-------------------------------------|
| Install deps          | `uv sync`                           |
| Add a package         | `uv add <package>`                  |
| Run Django commands   | `uv run manage.py <command>`        |
| Format code           | `uv run ruff format .`              |
| Check linting issues  | `uv run ruff check .`               |
| Fix linting issues    | `uv run ruff check . --fix`         |