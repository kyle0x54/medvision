import io

from Crypto.Cipher import AES


def _pad16(s, pad_ch="\0"):
    if isinstance(s, str):
        s = s.encode("utf-8")
    length = (16 - len(s) % 16) % 16
    s += (pad_ch * length).encode("utf-8")
    return s


def encrypt(data, key, iv, save_path=None):
    """Encrypt file or data.

    Args:
        data (boyes or str): Data or a file path.
        key (str or bytes): The secret key to use in the symmetric cipher.
        iv (str or bytes): The initialization vector to use for encryption or
            decryption.
        save_path (str): The save path of encrypted data.

    Returns:
        (bytes): The encrypted data.
    """
    if isinstance(data, str):
        with open(data, "rb") as f:
            data = f.read()
    length = str(len(data))
    length = _pad16(length)

    key = _pad16(key)
    iv = _pad16(iv)
    data = _pad16(data)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    data = cipher.encrypt(data)
    data = length + data
    if save_path:
        with open(save_path, "wb") as f:
            f.write(data)
    return data


def decrypt(data, key, iv, save_path=None):
    """Decrypt file or data.

    Args:
        data (boyes or str): Data or a file path.
        key (str or bytes): The secret key to use in the symmetric cipher.
        iv (str or bytes): The initialization vector to use for encryption or
            decryption.
        save_path (str): The save path of decrypted data.

    Returns:
        (bytes): The decrypted data.
    """
    if isinstance(data, str):
        with open(data, "rb") as f:
            data = f.read()
    pad_ch = "\0"
    length = int(data[:16].rstrip(pad_ch.encode("utf-8")).decode("utf-8"))
    data = data[16:]
    key = _pad16(key)
    iv = _pad16(iv)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    data = cipher.decrypt(data)
    data = data[:length]
    if save_path:
        with open(save_path, "wb") as f:
            f.write(data)
    return data


def decrypt_to_file_object(data, key, iv, save_path=None):
    """Decrypt a file or data and convert to file object.

    Args:
        data (boyes or str): Data or a file path.
        key (str or bytes): The secret key to use in the symmetric cipher.
        iv (str or bytes): The initialization vector to use for encryption or
            decryption.
        save_path (str): The save path of decrypted data.

    Returns:
        (file-like object): The decrypted data.
    """
    data = decrypt(data, key, iv, save_path)
    return io.BytesIO(data)
