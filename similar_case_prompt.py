SIMILAR_CASE_PROMPT = """
###  **Similar Case Details**
- Assign realistic **Plaintiff & Defendant** names  
- Assign a **Judge's name**  
- Assign **Lawyers for both sides**  
- Clearly explain the **charges or dispute**  

###  **Court Proceedings**
- **Opening Statements**:  
  - Plaintiff's Opening Statement: [Summary of plaintiff's claims and desired outcome].  
  - Defendant's Opening Statement: [Summary of defendant's defense and counterclaims].  

- **Presentation of Evidence**:  
  - Plaintiff's Evidence: [Detailed description of evidence and witness testimonies].  
  - Defendant's Evidence: [Detailed description of evidence and witness testimonies].  
  - Exhibits: [List and describe all exhibits submitted by both parties].  
  - Witness Testimonies: [Include direct examination questions and answers for each witness].  

- **Cross-Examinations**:  
  - Plaintiff's Cross-Examination: [Questions posed by defendant's lawyer to plaintiff's witnesses].  
  - Defendant's Cross-Examination: [Questions posed by plaintiff's lawyer to defendant's witnesses].  
  - Rebuttal Witnesses: [Additional witnesses called to counter points raised during cross-examination].  

- **Objections and Rulings**:  
  - [Include realistic objections and the judge's rulings with legal reasoning].  

- **Final Arguments**:  
  - Plaintiff's Closing Argument: [Persuasive summary of plaintiff's case].  
  - Defendant's Closing Argument: [Persuasive summary of defendant's case].  

- **Judge's Verdict**:  
  - Legal Reasoning: [Detailed explanation of the judge's decision].  
  - Case Law References: [Cite relevant Indian case laws or Supreme Court judgments].  
  - Judicial Precedents: [Discuss how similar cases were decided in the past].  

- **Punishment or Settlement**:  
  - Sentencing: [Describe the punishment and its legal basis].  
  - Compensation: [Detail any monetary compensation or restitution awarded].  
  - Appeal Process: [Mention the possibility of an appeal and its grounds].  

- **Courtroom Atmosphere**:  
  - [Describe the tone and demeanor of the judge, lawyers, and witnesses].  
  - [Include any dramatic or tense moments during the proceedings].  

- **Post-Trial Actions**:  
  - Compliance with Verdict: [Explain how the verdict will be enforced].  
  - Future Implications: [Discuss the broader impact of the verdict on similar cases].  

Ensure that the response is complete, realistic, and structured.
"""