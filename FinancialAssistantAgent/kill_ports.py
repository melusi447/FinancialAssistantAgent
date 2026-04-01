"""
Kill processes using ports 8000, 8001, 7860, 7861
"""

import subprocess
import sys
import os

def kill_port(port):
    """Kill process using a specific port"""
    try:
        # Find process using the port
        result = subprocess.run(
            f'netstat -ano | findstr :{port}',
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        try:
                            # Kill the process
                            subprocess.run(f'taskkill /PID {pid} /F', shell=True, check=True)
                            print(f"Killed process {pid} using port {port}")
                        except:
                            print(f"Could not kill process {pid} using port {port}")
                    break
        else:
            print(f"No process found using port {port}")
    except Exception as e:
        print(f"Error checking port {port}: {e}")

def main():
    """Kill processes using common ports"""
    ports = [8000, 8001, 7860, 7861, 8002, 8003, 7862, 7863]
    
    print("Killing processes using common ports...")
    for port in ports:
        kill_port(port)
    
    print("Done!")

if __name__ == "__main__":
    main()
