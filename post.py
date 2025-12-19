import imaplib
import email
from email.header import decode_header
import os

imap_server = "imap.qq.com"
email_user = ''
email_password = ''  

mail = imaplib.IMAP4_SSL(imap_server)
mail.login(email_user, email_password)
print("邮箱登录成功")

mail.select("inbox")

status, messages = mail.search(None, 'ALL')
mail_ids = messages[0].split()

latest_ids = mail_ids[-10:] if len(mail_ids) >= 10 else mail_ids

TARGET_FILES = {"中金香港日度交易总结.xlsx", "VAL-RuiDaInternatio-2585943_2585945.20250605.xlsx","1.xlsx"}

saved = set()

SAVE_DIR = os.path.join(os.getcwd(), "attachments")
os.makedirs(SAVE_DIR, exist_ok=True)

for mail_id in latest_ids:
  
    if saved == TARGET_FILES:
        break
        
    status, msg_data = mail.fetch(mail_id, "(RFC822)")
    raw_email = msg_data[0][1]
    email_message = email.message_from_bytes(raw_email)
    
    subject, encoding = decode_header(email_message["Subject"])[0]
    if isinstance(subject, bytes):
        subject = subject.decode(encoding or "utf-8")
    print(f"\n处理邮件：{subject}（ID: {mail_id.decode()}）")
    
   
    for part in email_message.walk():
       
        if part.get_content_maintype() == "multipart":
            continue
        
        if part.get("Content-Disposition") is None:
            continue
        
        raw_filename = part.get_filename()
        if raw_filename is None:
            continue  
        
        filename, encoding = decode_header(raw_filename)[0]
        if isinstance(filename, bytes):
            filename = filename.decode(encoding or "utf-8")
        
        if filename in TARGET_FILES and filename not in saved:
            # 保存附件
            save_path = os.path.join(SAVE_DIR, filename)
            with open(save_path, "wb") as f:
                f.write(part.get_payload(decode=True))  
            saved.add(filename)
            print(f"已保存附件：{save_path}")
            
            if saved == TARGET_FILES:
                print("所有目标附件已保存，退出程序")
                break  

mail.close()
mail.logout()
print("邮箱连接已关闭")
