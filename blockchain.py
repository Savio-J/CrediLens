"""
Blockchain Score Registry - FREE Tier Implementation
Uses Sepolia Testnet (Ethereum test network) for demo purposes.

Setup Requirements:
1. Sign up at https://infura.io (FREE)
2. Create a new project and get your API key
3. Add INFURA_API_KEY to your .env file
4. Get free Sepolia ETH from https://sepoliafaucet.com/
"""

import hashlib
import json
import os
from datetime import datetime
from web3 import Web3
from eth_account import Account
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SEPOLIA_RPC = "https://sepolia.infura.io/v3/{api_key}"

# Demo mode - simulates blockchain without real connection
DEMO_MODE = False  # Set to True for demo mode without real blockchain


def get_web3_connection():
    """Get Web3 connection to Sepolia testnet."""
    load_dotenv()  # Reload to get latest values
    api_key = os.environ.get("INFURA_API_KEY")
    
    if not api_key or DEMO_MODE:
        return None  # Demo mode
    
    try:
        w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC.format(api_key=api_key)))
        if w3.is_connected():
            return w3
    except Exception:
        pass  # Connection failed, return None
    
    return None


def generate_wallet():
    """Generate a new Ethereum wallet for the application."""
    account = Account.create()
    return {
        "address": account.address,
        "private_key": account.key.hex()
    }


def create_score_hash(product_id, product_name, score, specs_dict):
    """
    Create a SHA256 hash of product score data.
    This hash will be stored on blockchain as proof.
    """
    timestamp = datetime.utcnow().isoformat()
    
    # Create deterministic data string
    data = {
        "product_id": product_id,
        "product_name": product_name,
        "score": round(score, 2) if score else 0,
        "timestamp": timestamp,
        "specs_hash": hashlib.sha256(json.dumps(specs_dict, sort_keys=True).encode()).hexdigest()[:16]
    }
    
    # Create hash
    data_string = json.dumps(data, sort_keys=True)
    score_hash = hashlib.sha256(data_string.encode()).hexdigest()
    
    return {
        "hash": score_hash,
        "timestamp": timestamp,
        "data": data
    }


def store_hash_on_blockchain(score_hash, private_key=None):
    """
    Store score hash on Sepolia testnet.
    Returns transaction hash as proof.
    """
    w3 = get_web3_connection()
    
    if not w3:
        # DEMO MODE - Simulate blockchain storage
        # Generate a fake but realistic-looking tx hash
        fake_tx = hashlib.sha256(f"{score_hash}{datetime.utcnow().isoformat()}".encode()).hexdigest()
        return {
            "success": True,
            "tx_hash": f"0x{fake_tx}",
            "network": "demo",
            "block": None,
            "message": "Demo mode - hash stored locally (configure Infura for real blockchain)"
        }
    
    # REAL BLOCKCHAIN MODE
    try:
        wallet_address = os.environ.get("BLOCKCHAIN_WALLET")
        private_key = private_key or os.environ.get("BLOCKCHAIN_PRIVATE_KEY")
        
        if not wallet_address or not private_key:
            return {"success": False, "error": "Wallet not configured"}
        
        # Build transaction with hash in data field
        nonce = w3.eth.get_transaction_count(wallet_address)
        
        tx = {
            'nonce': nonce,
            'to': '0x0000000000000000000000000000000000000000',  # Null address (burn)
            'value': 0,
            'gas': 25000,
            'gasPrice': w3.eth.gas_price,
            'data': w3.to_hex(text=score_hash),
            'chainId': 11155111  # Sepolia chain ID
        }
        
        # Sign and send transaction
        signed = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        
        # Wait for confirmation
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        
        # Ensure tx_hash has 0x prefix for Etherscan
        tx_hash_hex = tx_hash.hex()
        if not tx_hash_hex.startswith('0x'):
            tx_hash_hex = '0x' + tx_hash_hex
        
        return {
            "success": True,
            "tx_hash": tx_hash_hex,
            "network": "sepolia",
            "block": receipt['blockNumber'],
            "message": "Hash stored on Sepolia testnet"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def verify_score_hash(product_id, product_name, claimed_score, specs_dict, stored_hash, stored_timestamp):
    """
    Verify that a score hasn't been tampered with.
    Recalculates hash and compares with stored value.
    """
    # Recreate the data structure
    data = {
        "product_id": product_id,
        "product_name": product_name,
        "score": round(claimed_score, 2) if claimed_score else 0,
        "timestamp": stored_timestamp,
        "specs_hash": hashlib.sha256(json.dumps(specs_dict, sort_keys=True).encode()).hexdigest()[:16]
    }
    
    # Recalculate hash
    data_string = json.dumps(data, sort_keys=True)
    recalculated_hash = hashlib.sha256(data_string.encode()).hexdigest()
    
    return {
        "verified": recalculated_hash == stored_hash,
        "stored_hash": stored_hash,
        "calculated_hash": recalculated_hash,
        "match": recalculated_hash == stored_hash
    }


def get_blockchain_explorer_url(tx_hash, network="sepolia"):
    """Get the URL to view transaction on block explorer."""
    if network == "demo":
        return None
    elif network == "sepolia":
        return f"https://sepolia.etherscan.io/tx/{tx_hash}"
    elif network == "mainnet":
        return f"https://etherscan.io/tx/{tx_hash}"
    return None


def get_blockchain_status():
    """Check if blockchain connection is active."""
    w3 = get_web3_connection()
    
    if not w3:
        return {
            "connected": False,
            "mode": "demo",
            "message": "Running in demo mode. Add INFURA_API_KEY to .env for real blockchain."
        }
    
    try:
        block = w3.eth.block_number
        return {
            "connected": True,
            "mode": "sepolia",
            "block_number": block,
            "message": f"Connected to Sepolia testnet (Block: {block})"
        }
    except Exception as e:
        return {
            "connected": False,
            "mode": "error",
            "message": str(e)
        }
