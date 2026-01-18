# deloris_ai/wallet.py
# [MODULE: WEB3 WALLET - FINANCIAL AUTONOMY]
# Giúp Deloris có tài sản riêng và thực hiện giao dịch on-chain.

import os
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()

class CryptoWallet:
    def __init__(self):
        # 1. Cấu hình mạng Blockchain
        # Mặc định dùng Sepolia Testnet nếu không có cấu hình khác
        self.rpc_url = os.getenv("WEB3_RPC_URL", "https://rpc.ankr.com/eth_sepolia")
        self.private_key = os.getenv("DELORIS_PRIVATE_KEY")
        
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        self.account = None
        self.is_connected = False

        self._connect_wallet()

    def _connect_wallet(self):
        if not self.w3.is_connected():
            print("⚠️ [WALLET] Không thể kết nối tới Blockchain Node.")
            return

        if not self.private_key:
            print("⚠️ [WALLET] Thiếu Private Key (Cần cấu hình trong .env). Chế độ Wallet: OFF.")
            return

        try:
            self.account = self.w3.eth.account.from_key(self.private_key)
            self.is_connected = True
            print(f"💰 [WALLET] Đã mở ví Deloris: {self.account.address}")
            print(f"   -> Network Chain ID: {self.w3.eth.chain_id}")
        except Exception as e:
            print(f"⚠️ [WALLET ERROR] Lỗi khóa bí mật: {e}")

    def get_balance(self):
        """Kiểm tra số dư ETH hiện tại"""
        if not self.account: return "Ví chưa kích hoạt."
        try:
            balance_wei = self.w3.eth.get_balance(self.account.address)
            balance_eth = self.w3.from_wei(balance_wei, 'ether')
            return f"{balance_eth:.4f} ETH"
        except Exception as e:
            return f"Lỗi đọc số dư: {e}"

    def send_eth(self, to_address, amount_eth):
        """
        Gửi tiền cho người khác (Deloris tự chi tiêu)
        """
        if not self.is_connected: return "Tôi không có quyền truy cập ví để gửi tiền."
        
        try:
            print(f"💸 [WALLET] Deloris đang gửi {amount_eth} ETH tới {to_address}...")
            
            # Kiểm tra địa chỉ hợp lệ
            if not self.w3.is_address(to_address):
                return "Địa chỉ ví người nhận không hợp lệ."
                
            to_address = self.w3.to_checksum_address(to_address)
            
            # Tạo giao dịch
            tx = {
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'to': to_address,
                'value': self.w3.to_wei(amount_eth, 'ether'),
                'gas': 21000,
                'gasPrice': self.w3.eth.gas_price,
                'chainId': self.w3.eth.chain_id
            }
            
            # Ký giao dịch
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            
            # Gửi lên mạng lưới
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            tx_hex = self.w3.to_hex(tx_hash)
            
            return f"✅ Giao dịch thành công!\nHash: `{tx_hex}`\n[Xem trên Explorer](https://sepolia.etherscan.io/tx/{tx_hex})"
            
        except Exception as e:
            print(f"❌ [WALLET ERROR] Giao dịch thất bại: {e}")
            return f"Giao dịch thất bại: {str(e)}"

    def get_address(self):
        return self.account.address if self.account else "Chưa thiết lập ví"