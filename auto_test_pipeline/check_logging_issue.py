#!/usr/bin/env python3
"""
簡單測試 train_lora.py 是否包含 logging_interval 參數
"""

def check_for_logging_interval():
    """檢查 train_lora.py 中是否有 logging_interval"""
    try:
        with open('train_lora.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'logging_interval' in content:
            print("❌ 發現 logging_interval 參數！")
            # 找出具體位置
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if 'logging_interval' in line:
                    print(f"   第 {i} 行: {line.strip()}")
            return False
        else:
            print("✅ train_lora.py 中沒有 logging_interval 參數")
            return True
    except Exception as e:
        print(f"❌ 讀取文件時出錯: {e}")
        return False

def check_train_network_params():
    """檢查 train_network.py 支援的參數"""
    import subprocess
    try:
        result = subprocess.run(['python', 'train_network.py', '--help'], 
                              capture_output=True, text=True, encoding='utf-8')
        
        if 'logging_interval' in result.stdout:
            print("✅ train_network.py 支援 logging_interval 參數")
            return True
        else:
            print("❌ train_network.py 不支援 logging_interval 參數")
            return False
    except Exception as e:
        print(f"❌ 檢查 train_network.py 參數時出錯: {e}")
        return False

if __name__ == "__main__":
    print("🔍 檢查 logging_interval 參數問題...")
    print()
    
    print("1. 檢查 train_lora.py:")
    lora_ok = check_for_logging_interval()
    
    print()
    print("2. 檢查 train_network.py 支援的參數:")
    network_ok = check_train_network_params()
    
    print()
    if lora_ok and not network_ok:
        print("🎯 問題確認: train_lora.py 沒有問題，train_network.py 不支援 logging_interval")
        print("💡 解決方案: 移除 logging_interval 相關參數")
    elif not lora_ok:
        print("🎯 問題確認: train_lora.py 包含不支援的 logging_interval 參數")
        print("💡 解決方案: 從 train_lora.py 移除 logging_interval 參數")
    else:
        print("🎯 沒有發現明顯問題")
