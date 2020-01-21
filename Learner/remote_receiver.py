import zmq
import signal
import sys
import getopt
from learners.efsm_learner import GrammarLearner

# tasks_config.challenge.json -l learners.base.RemoteLearner --learner-cmd "python remote_receiver.py"

def receive(socket):
    reply = socket.recv()
    if type(reply) == type(b''):
        reply = reply.decode('utf-8')
    return reply

def main():
    host = '127.0.0.1' #'172.18.0.1'
    port = 5556

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'a:p:', ['address=', 'port='])
    except getopt.GetoptError:
        print("Error reading options. Usage:")
        print("  example_learner.py [-a <address> -p <port>]")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-a', '--address'):
            host = arg
        elif opt in ('-p', '--port'):
            try:
                port = int(arg)
                if port < 1 or port > 65535:
                    raise ValueError("Out of range")
            except ValueError:
                print("Invalid port number: %s" % arg)
                sys.exit(2)

    context = zmq.Context()
    socket = context.socket(zmq.PAIR)

    address = "tcp://%s:%s" % (host, port)
    print("Connecting to %s" % address)
    sys.stdout.flush()

    socket.connect(address)
    socket.send_string("hello")


    def handler(signal, frame):
        print('exiting...')
        socket.disconnect(address)
        exit()


    signal.signal(signal.SIGINT, handler)

    learner = GrammarLearner()

    reward = receive(socket)
    
    while True:
        input = receive(socket)
        output = learner.next(input)
        socket.send_string(output)  # your attempt to solve the current task
        reward = receive(socket)
        learner.reward(int(reward))
        
    signal.pause()

if __name__ == '__main__':
    main()
