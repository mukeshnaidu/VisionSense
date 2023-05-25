from twilio.rest import Client

account_sid = 'ACa5e43dc25802687dfe14ac2df99bf5f4'
auth_token = '0b22b5f49f0c0e33412b0ea48edd7ed7'
client = Client(account_sid, auth_token)

textMessage = "Today, we have achieved an 73% accuracy rate in predicting the footfall, which amounts to 3,230 visitors. We anticipate the crowd to be more active between 11 AM and 12 PM and between 5 PM and 9 PM. To ensure effective staff management, please be prepared to allocate appropriate resources during these time frames."

message = client.messages.create(
  from_='whatsapp:+14155238886',
  body=textMessage,
  to='whatsapp:+971529123587'
)

print(message.sid)