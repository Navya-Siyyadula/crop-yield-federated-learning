# Update security

The update wrapper uses AES with a 256-bit key in Galois/Counter Mode (AES-256-GCM). It encrypts serialized model arrays and authenticates ciphertext integrity. A fresh random nonce is prepended to each encrypted payload; the AEAD implementation appends the authentication tag.

In the final Flower flow, clients encrypt local model updates before sending them. The server authenticates and decrypts each update for FedAvg, then encrypts the aggregated global weights before clients receive them. Successful decryption verifies the authentication tag; tampered ciphertext is rejected. The same 32-byte experiment key is shared by the four local clients and server. This local key arrangement is not a production multi-party key-management design.

All clients train in the same normalized target space using one scaler fitted on the complete 350-row training target set. Scaler parameters are configuration values, not encrypted model updates; no validation or test target is used to fit them.

This is an adaptation of the referenced base-paper architecture. The repository does not implement the base paper's AES/RSA/AES key-management arrangement, and RSA must not be claimed. Key distribution and rotation remain deployment concerns; an environment-loaded shared key is not a complete multi-party key-management protocol. Encryption does not hide sample counts or timing metadata.
