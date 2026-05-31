"""Run commands on remote server via paramiko SSH.

Set the following environment variables before use:
    SSH_HOST, SSH_PORT, SSH_USER, SSH_PASS
"""
import os
import paramiko
import sys

HOST = os.environ.get("SSH_HOST", "")
PORT = int(os.environ.get("SSH_PORT", 22))
USER = os.environ.get("SSH_USER", "root")
PASS = os.environ.get("SSH_PASS", "")

def ssh_cmd(command, timeout=120):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    client.close()
    return out, err

def sftp_put(local_path, remote_path):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
    sftp = client.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()
    client.close()
    print(f"  Uploaded: {local_path} -> {remote_path}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "echo 'SSH OK' && nvidia-smi --query-gpu=name --format=csv,noheader"
    out, err = ssh_cmd(cmd)
    print(out)
    if err:
        print("STDERR:", err)
