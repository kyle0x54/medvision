import io
from Crypto.Cipher import AES


def _pad16(s, pad_ch='\0'):
    if isinstance(s, str):
        s = s.encode('utf-8')
    length = (16 - len(s) % 16) % 16
    s += (pad_ch * length).encode('utf-8')
    return s


def encrypt(data, key, iv, save_path=None):
    """ Encrypt file or data.

    data (boyes or str): Data or a file path.
    key (str or bytes): The secret key to use in the symmetric cipher.
    iv (str or bytes): The initialization vector to use for encryption or decryption.
    save_path (str): The save path of encrypted data.

    return (bytes): The encrypted data.
    """
    if isinstance(data, str):
        with open(data, 'rb') as f:
            data = f.read()
    pad_ch = '\0'
    key = _pad16(key, pad_ch)
    iv = _pad16(iv, pad_ch)
    data = _pad16(data, pad_ch)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    data = cipher.encrypt(data)
    if save_path:
        with open(save_path, 'wb') as f:
            f.write(data)
    return data


def decrypt(data, key, iv, save_path=None):
    """ Decrypt file or data.

    data (boyes or str): Data or a file path.
    key (str or bytes): The secret key to use in the symmetric cipher.
    iv (str or bytes): The initialization vector to use for encryption or decryption.
    save_path (str): The save path of decrypted data.

    return (bytes): The decrypted data.
    """
    if isinstance(data, str):
        with open(data, 'rb') as f:
            data = f.read()
    pad_ch = '\0'
    key = _pad16(key, pad_ch)
    iv = _pad16(iv, pad_ch)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    data = cipher.decrypt(data)
    data = data.rstrip(pad_ch.encode('utf-8'))
    if save_path:
        with open(save_path, 'wb') as f:
            f.write(data)
    return data


def decrypt_to_file_object(data, key, iv, save_path=None):
    """ Decrypt a file or data and convert to file object.

    data (boyes or str): Data or a file path.
    key (str or bytes): The secret key to use in the symmetric cipher.
    iv (str or bytes): The initialization vector to use for encryption or decryption.
    save_path (str): The save path of decrypted data.

    return (file-like object): The decrypted data.
    """
    data = decrypt(data, key, iv, save_path)
    return io.BytesIO(data)
