1. 进入仓库目录
首先确保你的路径定位到 test 文件夹：

Bash
cd /e/BNU/OneDrive/BNU/code/test
2. 检查当前状态（可选）
输入这行可以看到哪些文件被修改或新增了：

Bash
git status
你应该能看到 0-MP-SPDZ/ 显示为红色（未追踪的文件）。

3. 添加、提交并推送
按照 Git 的标准流程操作：

Bash
# 1. 将所有新增的文件添加到暂存区
git add .

# 2. 记录这次上传，双引号里可以写你的备注
git commit -m "上传 MP-SPDZ 项目代码"

# 3. 将代码推送到 GitHub
git push origin main
注：如果报错提示分支不对，请试着把最后一行改为 git push origin master。