import os
import subprocess
import sys
import time

mode = sys.argv[1] if len(sys.argv) > 1 else 'stop'

def play_sound():
    subprocess.run(
        ['powershell', '-c', '[System.Media.SystemSounds]::Exclamation.Play(); Start-Sleep -Milliseconds 500'],
        capture_output=True
    )

def show_popup(msg):
    ps = f'$shell = New-Object -ComObject WScript.Shell; $shell.Popup("{msg}", 0, "Claude Code", 64)'
    subprocess.run(['powershell', '-c', ps], capture_output=True)

if mode == 'pretool':
    # Bash tool requested — might need confirmation, alert the user
    play_sound()

else:  # stop mode
    tmp = os.environ.get('TEMP', os.environ.get('TMP', 'C:/Windows/Temp'))
    f = os.path.join(tmp, '.claude_task_start')
    if os.path.exists(f):
        elapsed = time.time() - float(open(f).read().strip())
        os.remove(f)
        if elapsed > 15:
            show_popup(f'Claude finished! ({int(elapsed)}s)')
        elif elapsed > 3:
            # Planning done or quick confirmation request — just a sound
            play_sound()
