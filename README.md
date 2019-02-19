# Project Proposal

Using natural language processing to examine the federal budget

# Background

The idea for this project came from an [article I saw](https://www.texastribune.org/2019/02/14/government-shutdown-deal-includes-protections-south-texas-landmarks/) which mentioned that the most recent budget bill attempts to spare the 
National Butterfly Reserve from being destroyed by border wall construction projects. 

This led to the idea that 'butterfly' is probably an unusual word for a budget bill. 

What other unusual sections might be lurking in the 400+ page Federal budget? 

# Need for study

Slipping extra riders into 'must pass' bills is a common legislative tactic. A machine learning model that can identify 
unusual language or provisions in a bill would be useful for:

- Congressional staffers
- Political activists
- Lobbyists
- Concerned citizens

# Data

The dataset is H.J. Res 31 (Enrolled Bill), the bill which funds the government through September, 2019. The PDF vesion of this bill is 465 pages long. I plan to work with the [XML file](https://www.congress.gov/116/bills/hjres31/BILLS-116hjres31enr.xml)
which may be more amenable to automated scanning. 

If there is time, I may also include earlier drafts and conference reports of the bill to enable the model to focus specifically on unusual items added in the final hours before voting.  
