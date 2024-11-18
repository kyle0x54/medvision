import ftplib
import logging
import os
import socket
import sys


class FTP:
    ftp = ftplib.FTP()

    def __init__(self, host, port=21):
        self.host = host
        self.ftp.connect(host, port)

    def login(self, user, passwd):
        try:
            self.ftp.login(user, passwd)
            logging.info("Connection established: [%s]", self.host)
        except (socket.error, socket.gaierror):
            logging.error(
                "FTP login failed, please check correctness of "
                "host machine ip, user name and password."
            )
            sys.exit(0)

    def logout(self):
        logging.info("Disconnected from server: [%s]", self.host)
        self.ftp.quit()

    def download_file(self, local_file, remote_file):
        file_handler = open(local_file, "wb")
        logging.info(file_handler)
        self.ftp.retrbinary("RETR " + remote_file, file_handler.write)
        file_handler.close()
        logging.info("[%s] File transfer successful.", remote_file)
        return True

    def is_ftp_dir(self, file_name):
        try:
            self.ftp.cwd(file_name)
            self.ftp.cwd("..")
            return True
        except ftplib.error_perm:
            return False

    def download_tree(self, local_dir, remote_dir):
        logging.info("remote_dir, %s", remote_dir)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        self.ftp.cwd(remote_dir)
        remote_names = self.ftp.nlst()
        logging.info("remote_names, %s", remote_names)
        for file in remote_names:
            local = os.path.join(local_dir, file)
            logging.info(self.ftp.nlst(file))
            if self.is_ftp_dir(file):
                if not os.path.exists(local):
                    os.makedirs(local)
                self.download_tree(local, file)
            else:
                self.download_file(local, file)

        self.ftp.cwd("..")
        return


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    local_dir = "test_data"
    remote_dir = "/release-dependencies/"

    HOST_216 = "192.168.1.216"
    USER_216 = "reader"
    PASSWD_216 = "reader"

    ftp216 = FTP(HOST_216)
    ftp216.login(user=USER_216, passwd=PASSWD_216)
    ftp216.download_tree(local_dir, remote_dir)
    ftp216.logout()
