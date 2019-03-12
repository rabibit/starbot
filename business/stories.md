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

## happy path 1
* greet
    - utter_greet
* book_room
    - room_form
    - form{"name": "room_form"}
    - form{"name": null}
    - utter_done

## happy path 2
* book_room
    - room_form
    - form{"name": "room_form"}
    - form{"name": null}
    - utter_done

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

## consultation path
* consultation
    - utter_introduction
    
## room_available path
* room_available
    - utter_room_idle
    
## room_price path
* room_price
    - utter_room_price

