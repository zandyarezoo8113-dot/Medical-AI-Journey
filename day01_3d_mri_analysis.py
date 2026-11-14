import numpy as np

# 1. شبیه‌سازی یک اسکن سه‌بعدی MRI مغز شامل ۵ برش (Slice) با ابعاد ۱۰ در ۱۰
# اعداد بین ۱۰ تا ۵۰ نشان‌دهنده بافت سالم مغز هستند
np.random.seed(42)
brain_mri_3d = np.random.randint(10, 50, size=(5, 10, 10))

# 2. ایجاد یک تومور مصنوعی با شدت روشنایی بالا (۲۲۰ تا ۲۵۰) فقط در برش چهارم (اندیس ۳)
brain_mri_3d[3, 4:7, 4:7] = np.random.randint(220, 250, size=(3, 3))

print(f"ابعاد کل تنسور اسکن سه‌بعدی: {brain_mri_3d.shape}")

# 3. بررسی تک‌تک برش‌ها برای پیدا کردن تومور با حد آستانه (Threshold = 200)
# نکته اصلاحی: با [0] اولین عدد ابعاد (یعنی تعداد برش‌ها = 5) را به range می‌دهیم
for slice_idx in range(brain_mri_3d.shape[0]):
    current_slice = brain_mri_3d[slice_idx]
    
    # تشخیص پیکسل‌های مشکوک بالای ۲۰۰
    tumor_mask = current_slice > 200
    tumor_pixels = np.sum(tumor_mask)
    
    if tumor_pixels > 0:
        print(f"🚨 هشدار: تومور در برش شماره {slice_idx + 1} (اندیس {slice_idx}) شناسایی شد! مساحت: {tumor_pixels} پیکسل")
    else:
        print(f"✅ برش شماره {slice_idx + 1}: بافت سالم است.")

# 4. نرمال‌سازی کل اسکن سه‌بعدی به بازه ۰ تا ۱ برای ورود به مدل‌های یادگیری عمیق
normalized_scan = brain_mri_3d / 255.0
print("\nنرمال‌سازی تنسور سه‌بعدی با موفقیت انجام شد.")
