echo "Restarting backend..."
pkill -f "python3 main.py"
pkill -f "uvicorn"
nohup python3 main.py > backend.log 2>&1 &
echo "Done!"
