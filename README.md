<div align="center">

tGBM; model comparison and learner observations | خط الأساس وXGBoost وLightGBM؛ مقارنة النماذج وملاحظات المتدرب | [Colab](https://colab.research.google.com/github/almiyead-rgb/sda-dsc-211-student-template/blob/main/notebooks/01_baseline_boosting.ipynb) · [Guide](DAY1_GUIDE.md) |
| 2 | Honest validation, leakage control and bounded Optuna search; validation evidence | التحقق الصادق ومنع التسرب والبحث المحدود بـOptuna؛ أدلة التحقق | [Colab](https://colab.research.google.com/github/almiyead-rgb/sda-dsc-211-student-template/blob/main/notebooks/02_validation_tuning.ipynb) · [Guide](DAY2_GUIDE.md) |
| 3 | Imbalance, OOF probabilities, simulated decision cost and capacity; Decision Card | عدم التوازن واحتمالات OOF وتكلفة القرار التعليمية والسعة؛ بطاقة القرار | [Colab](https://colab.research.google.com/github/almiyead-rgb/sda-dsc-211-student-template/blob/main/notebooks/03_cost_sensitive_decision.ipynb) · [Guide](DAY3_GUIDE.md) |
| 4 | Permutation importance, SHAP, calibration and stability; interpretation report | أهمية التبديل وSHAP والمعايرة والاستقرار؛ تقرير التفسير | [Colab](https://colab.research.google.com/github/almiyead-rgb/sda-dsc-211-student-template/blob/main/notebooks/04_explain_calibrate.ipynb) · [Guide](DAY4_GUIDE.md) |
| 5 | Averaging, stacking, worth-it decision, Model Card and final package | المتوسطات والتكديس وقرار الجدوى وبطاقة النموذج والحزمة النهائية | [Colab](https://colab.research.google.com/github/almiyead-rgb/sda-dsc-211-student-template/blob/main/notebooks/05_final_model.ipynb) · [Guide](DAY5_GUIDE.md) |

## Repository map | خريطة المستودع

<table>
<tr>
<td width="50%" valign="top" dir="ltr">

- `notebooks/` — executed daily notebooks
- `artifacts/` — CSV, JSON, figures and model evidence
- `reports/` — Decision Card, interpretation report, ensemble decision and Model Card
- `submission/` — final predictions and manifest
- `presentation/` — five-slide presentation and final PDF
- `tamweel/` — reproducible inference interface
- `scripts/` — setup, checks and rebuild tools
- `data/` — synthetic course data and data contract

</td>
<td width="50%" valign="top" dir="rtl">

- `notebooks/` — دفاتر الأيام المنفذة
- `artifacts/` — ملفات CSV وJSON والرسوم وأدلة النموذج
- `reports/` — بطاقة القرار وتقرير التفسير وقرار التجميع وبطاقة النموذج
- `submission/` — التنبؤات النهائية والـManifest
- `presentation/` — قالب العرض والعرض النهائي من خمس شرائح
- `tamweel/` — واجهة استدلال قابلة لإعادة الإنتاج
- `scripts/` — أدوات الإعداد والفحص وإعادة البناء
- `data/` — بيانات الدورة الاصطناعية وعقد البيانات

</td>
</tr>
</table>

## Assessment | التقييم

<table>
<tr>
<td width="50%" valign="top" dir="ltr">

- **Project technical and administrative requirements:** 90 points
- **Presentation and discussion:** 10 points
- **Pass:** 70 / 100
- **Distinction:** 95 / 100

Read [RUBRIC.md](RUBRIC.md). High model performance alone is not enough; validation quality, reproducibility, interpretation and decision reasoning are assessed.

</td>
<td width="50%" valign="top" dir="rtl">

- **المتطلبات التقنية والإدارية للمشروع:** 90 درجة
- **العرض والمناقشة:** 10 درجات
- **النجاح:** 70 من 100
- **التميز:** 95 من 100

اقرأ [RUBRIC.md](RUBRIC.md). لا يكفي ارتفاع أداء النموذج؛ يُقيّم التحقق وقابلية إعادة الإنتاج والتفسير ومنطق القرار.

</td>
</tr>
</table>

## Final verification and submission | الفحص والتسليم النهائي

<table>
<tr>
<td width="50%" valign="top" dir="ltr">

1. Complete all learner responses and reports.
2. Run [Notebook 99](https://colab.research.google.com/github/almiyead-rgb/sda-dsc-211-student-template/blob/main/notebooks/99_final_submission_check.ipynb).
3. Correct issues in the original files and rerun the check.
4. Run **Actions → Final Project Check** in your repository.
5. Create a final tag on the exact checked commit.
6. Submit the repository URL, final tag, full commit SHA, check URL and presentation through the private cohort channel.

Technical readiness is not a grade or a submission receipt.

</td>
<td width="50%" valign="top" dir="rtl">

1. أكمل جميع إجابات المتدرب والتقارير.
2. شغّل [دفتر 99](https://colab.research.google.com/github/almiyead-rgb/sda-dsc-211-student-template/blob/main/notebooks/99_final_submission_check.ipynb).
3. صحح الملاحظات في الملفات الأصلية ثم أعد الفحص.
4. شغّل **Actions → Final Project Check** داخل مستودعك.
5. أنشئ Tag نهائيًا على Commit نفسه الذي اجتاز الفحص.
6. أرسل رابط المستودع وTag النهائي وCommit SHA الكامل ورابط الفحص والعرض عبر القناة الخاصة للدفعة.

الجاهزية التقنية ليست درجة ولا إيصال استلام.

</td>
</tr>
</table>

## Guides and support | الأدلة والدعم

| Need | Resource | الاحتياج | المرجع |
|---|---|---|---|
| Colab | [COLAB_GUIDE.md](COLAB_GUIDE.md) | استخدام Colab | [دليل Colab](COLAB_GUIDE.md) |
| GitHub | [GITHUB_GUIDE.md](GITHUB_GUIDE.md) | حفظ المشروع في GitHub | [دليل GitHub](GITHUB_GUIDE.md) |
| Requirements | [TECHNICAL_REQUIREMENTS.md](TECHNICAL_REQUIREMENTS.md) · [ADMINISTRATIVE_REQUIREMENTS.md](ADMINISTRATIVE_REQUIREMENTS.md) | المتطلبات | [التقنية](TECHNICAL_REQUIREMENTS.md) · [الإدارية](ADMINISTRATIVE_REQUIREMENTS.md) |
| Submission | [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) · [FINAL_CHECK_GUIDE.md](FINAL_CHECK_GUIDE.md) | التسليم والفحص | [دليل التسليم](SUBMISSION_GUIDE.md) · [دليل الفحص](FINAL_CHECK_GUIDE.md) |
| Terms | [GLOSSARY.md](GLOSSARY.md) | المصطلحات | [القاموس](GLOSSARY.md) |
| Troubleshooting | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) · [FAQ.md](FAQ.md) | حل المشكلات | [حل المشكلات](TROUBLESHOOTING.md) · [الأسئلة الشائعة](FAQ.md) |
| Learning resources | [LEARNING_RESOURCES.md](LEARNING_RESOURCES.md) | الفيديوهات والمراجع | [الموارد التعليمية](LEARNING_RESOURCES.md) |

> Do not place personal data, passwords, access tokens, private grades or submission receipts in a public repository.  
> لا تضع بيانات شخصية أو كلمات مرور أو رموز وصول أو درجات خاصة أو إيصالات تسليم داخل مستودع عام.

Prepared and delivered by **Meaad Al-Marri | ميعاد المري** · [Attribution and educational use](NOTICE.md).

The published learner release remains `v1.0.0`. This branch is the controlled bilingual candidate for `v1.1.0`.
