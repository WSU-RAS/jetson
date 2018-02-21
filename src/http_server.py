#!/usr/bin/env python2
"""
Very simple HTTP server

This potentially will be used for the tablet interface, where we open the
webpage on the tablet to display videos, prompts, etc.
"""
import os
import rospy
import socket
import SimpleHTTPServer
import SocketServer

class ReuseTCPServer(SocketServer.TCPServer):
    """
    Reuse the port so that when we kill it we don't have to wait 4 minutes
    before we can use the port again

    https://stackoverflow.com/a/18858817/2698494
    """
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)

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
        self.httpd = ReuseTCPServer(("", port), Handler)
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
        self.httpd.server_close()
        #self.httpd.shutdown()

if __name__ == '__main__':
    try:
        node = Server()
        node.run()
    except rospy.ROSInterruptException:
        pass
