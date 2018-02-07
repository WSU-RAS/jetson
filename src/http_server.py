#!/usr/bin/env python2
import os
import rospy
import SimpleHTTPServer
import SocketServer

class Server:
    def __init__(self):
        self.shutdown = False

        # Get parameters
        rospy.init_node('http_server')
        port = rospy.get_param("~http_port", 8080)
        directory = rospy.get_param("~directory", ".")

        # Switch to directory to serve
        os.chdir(directory)

        # Server
        Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
        self.httpd = SocketServer.TCPServer(("", port), Handler)
        rospy.loginfo("Serving HTTP at port "+str(port))
        rospy.on_shutdown(self.shutdown_hook)

    def run(self):
        try:
            self.httpd.serve_forever()
        except:
            if not self.shutdown:
                rospy.logerr("Error running HTTP server")

    def shutdown_hook(self):
        rospy.loginfo("Trying to shutdown server")
        self.shutdown = True # So we don't throw error on killing
        self.httpd.socket.close()
        #self.httpd.shutdown()

if __name__ == '__main__':
    try:
        node = Server()
        node.run()
    except rospy.ROSInterruptException:
        pass
