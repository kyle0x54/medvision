import io
from Crypto.Cipher import AES


def _pad16(s, pad_ch='\0'):
    if isinstance(s, str):
        s = s.encode('utf-8')
    length = (16 - len(s) % 16) % 16
    s += (pad_ch * length).encode('utf-8')
    return s


def encrypt(data, key, iv, save_path=None):
    """
    encryption.
    :param data:bytes(data) or str(a file path).
    :param key: str or bytes.
    :param iv: str or bytes.
    :return: bytes.
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


def decrypt(data, key, iv,
            save_path=None,
            return_fileObj=True):
    """
    decryption.
    :param data: bytes(data) or str(a file path).
    :param key: str or bytes.
    :param iv: str or bytes.
    :return: a file-like object if return_fileObj is True, else bytes.
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
    if return_fileObj:
        data = io.BytesIO(data)
    return data
