# Data Dictionary: RTW Prediction Model (Issue #5)

| Column Name | Description |
| :--- | :--- |
| **Claim No** | Unique identifier for the insurance claim. |
| **Occurrence No** | Sequence number of the incident. |
| **Claim Financial Year** | The fiscal year the claim was lodged. |
| **Agency NN** | Numeric identifier for the employer agency. |
| **Incident Date** | The date the injury occurred. |
| **Finalised Date** | The date the claim was officially settled/closed. |
| **Total Paid** | Total financial compensation paid on the claim. |
| **Paid Days Lost** | Total days of work lost that were compensated. |
| **Injury Agency Group** | High-level source of the injury (e.g., Machinery). |
| **Bodily Location Group** | Part of the body affected (e.g., Upper Limbs). |
| **Mechanism Group** | How the injury happened (e.g., Falls, Slips). |
| **Nature Group** | Medical category of injury (e.g., Sprains, Strains). |
| **Occupation** | The worker's specific job title. |
| **Gender** | Gender of the claimant (M/F). |
| **Date of Birth** | Claimant's date of birth. |
| **Days to RTW** | **(Target)** Number of days until return to work. |
| **RTW Category** | Categorical status of return (e.g., RTW 13 Weeks). |
| **Age at Accident Date** | Age of the worker at the time of the incident. |
| **Settled** | Settlement status indicator (Yes/No). |