@echo off
REM 將 lora_train_set 下所有圖片和 txt 搬到 myset 子資料夾
move lora_train_set\*.jpg lora_train_set\myset\
move lora_train_set\*.JPG lora_train_set\myset\
move lora_train_set\*.png lora_train_set\myset\
move lora_train_set\*.jpeg lora_train_set\myset\
move lora_train_set\*.txt lora_train_set\myset\
echo 搬移完成！
pause
