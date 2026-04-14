# Submodules (chat, suggestions, …) are imported explicitly by main.py.
# Avoid eager imports here so lightweight tests can `import app.routers.suggestions`
# without loading the entire application graph.
