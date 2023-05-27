from subprocess import *
import time

if __name__ == '__main__':
    p1 = Popen(['python', './ODESR_Generator_V2.py'])
    p2 = Popen(['python', './Finding_Frequencies_GUI.py'])
    try:
        time.sleep(3)
        while True:
            stop = str(input('[+] Do you want to stop [y/n] : '))
            if stop == 'y':
                p1.kill()
                p2.kill()
                break
    except KeyboardInterrupt:
        p1.kill()
        p2.kill()
        print('Press Ctrl-C to kill the program')
    