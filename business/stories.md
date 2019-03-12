## happy path
* greet
  - utter_greet

## say goodbye
* bye
  - utter_bye

 ## happy path
* greet
    - utter_greet
* book_room
    - room_form
    - form{"name": "room_form"}
    - form{"name": null}
    - utter_done
* thankyou
    - utter_noworries

## unhappy path
* greet
    - utter_greet
* book_room
    - room_form
    - form{"name": "room_form"}
* chitchat
    - utter_chitchat
    - room_form
    - form{"name": null}
    - utter_done
* thankyou
    - utter_noworries

## very unhappy path
* greet
    - utter_greet
* book_room
    - room_form
    - form{"name": "room_form"}
* chitchat
    - utter_chitchat
    - room_form
* chitchat
    - utter_chitchat
    - room_form
* chitchat
    - utter_chitchat
    - room_form
    - form{"name": null}
    - utter_done
* thankyou
    - utter_noworries

