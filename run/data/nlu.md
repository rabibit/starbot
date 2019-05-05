## intent:greet
- 你好
- hello
- hi
- 嘿
- 哈喽
- 哈罗
- 早上好
- 下午好
- good morning
- good afternoon
- 中午好


## intent:bye
? 您的房间已预留，请问还有其他需要么？

- bye
- goodbye
- 88
- 再见
- 晚安
- 拜拜
- 再会
- 慢走
- 明天见
- 一会儿再聊


## intent:ok
? 您看有问题吗？
? 对吗？

- 是的
- 对
- 可以
- 可以的
- 没错
- 好吧
- 好
- 行
- 没问题
- 就这样吧
- OK
- 👌
- yes


## intent:no
? 您看有问题吗？
? 对吗？

- 不
- 不行
- 不对
- 不用
- 不要
- 不不不
- 不要这样
- no
- 错了
- oh，no


## intent:thankyou
? 您的房间已预留，请问还有其他需要么？

- 谢谢
- 好的谢谢
- 谢了
- 多谢
- 非常感谢
- 太感谢了
- 辛苦了


## intent:book_room
@room_type: 房间|房|标间|单人间|大床房|豪华大床房|总统套房
@checkin_time: 今晚|明晚|今天|后天|周末|周六|一号|二号|十一号|二十号|二号下午|后天下午

? 您好，这里是未来酒店，请问有什么需要呢?
? 您的房间已预留，请问还有其他需要么？

- 我想订一个[房间](room_type)
- 我想订一间[房](room_type)
- 我想订一[标间](room_type)
- 我想订一个[标间](room_type)
- 帮我订一个[大床房](room_type)
- 可以帮我订一间[大床房](room_type)吗
- 我想订个[大床房](room_type),[今晚](checkin_time)入住
- 我想预订个[大床房](room_type)
- 我打算预订个[豪华套房](room_type)
- [今天](checkin_time)入住,还有空房间吗
- 我预订一间[大床房](room_type)豪华套房
- 行，那就2间[大床房](room_type)
- 两间[大床房](room_type)就可以了
- 我要三个[大床房](room_type)
- 那就预订[一](rooms)间商务标间吧
- 我觉得[两](rooms)间标间就足够了
- 我打算预订个[豪华套房](room_type)
- 我帮我朋友预订个[豪华套房](room_type)
- 我想订一个[房间](room_type)



@guest_name: 张三|李四|王麻子|杨康|穆念慈|郭啸天|李萍|杨铁心|包惜弱|完颜洪烈|柯镇恶|朱聪|韩宝驹|南希仁|张阿生|全金发|韩小莹|刘女士|王先生
@guest_phone_number: 123456|0281234343|02312334345

## intent:info
? 我们有标准间和大床房, 请问您需要哪种房间？
- [房间](room_type)

## intent:info
? 好的，请问怎么称呼您？

- [郭靖](guest_name)


## intent:info
? 您的联系电话?

- [18088888888](guest_phone_number)
- [15988888888](guest_phone_number)是我的电话号码
- 号码是[13188888888](guest_phone_number)
- 我的电话号码是[13188888888](guest_phone_number)
- 这是我的电话号码[13188888888](guest_phone_number)


## intent:info
? 您什么时候入住呢?

- [今晚](checkin_time)
- 我估计[今晚](checkin_time)入住
- [3月18号](checkin_time)
- 我打算[今晚](checkin_time)住进来
- 我大概[今晚](checkin_time)到达
- [周五](checkin_time)
- [下周三](checkin_time)


## intent:info
? 好的，请问怎么称呼您？

- 我叫[杰哥](guest_name)
- 我是[王先生](guest_name)
- 我的名字是[杰哥](guest_name)
- [郭靖](guest_name)
- [黄蓉](guest_name)


## intent:confirm
? 您看有问题吗？
? 对吗？

- 对的
- 没错
- 没问题
- 是的
- 对
- 没错订了
- ok
- yes
- 好的


## intent:change_info
@wrong_time: =checkin_time
@wrong_name: =guest_name

? 您看有问题吗？
? 对吗？

- 不是[今天](wrong_time)，是[明天](checkin_time)
- 不是[张三](wrong_name)，是[李四](guest_name)
- 名字不对，应该是[大锤](guest_name)
- 时间错了，是[后天下午](checkin_time)
- 不是[今天](wrong_time)是[明天](checkin_time)
- 不是[张三](wrong_name)是[李四](guest_name)
- 不是叫[张三](wrong_name)，我叫[李四](guest_name)
- 不是[张三](wrong_name)我叫[李四](guest_name)
- 我不叫[张三](wrong_name)我叫[李四](guest_name)


## intent:consultation
? 您好，这里是未来酒店，请问有什么需要呢?

- 请问你们的酒店环境怎么样
- 能简单介绍一下你们的酒店吗
- 你们的特色是什么
- 我是外地人，听朋友说你们酒店很好，可以向我介绍一下吗
- 我向了解一下你们的酒店
- 麻烦介绍一下你们的酒店
- 那你介绍介绍你们的酒店有什么特色呢


## intent:room_price
- 那你们的房价是什么样的呢
- 我想了解一下价格
- 价格怎样
- 房间价格如何
- 能说说房间价格么
- 那房间价格怎么样


## intent:room_available
- 还有空房间吗
- 我明天入住，还有房间吗
- 有标间吗
- 有大床房吗
- 有商务套房吗
- 有豪华套房吗


## intent:enha
- 那不错
- 好棒
- 我喜欢
- 非常棒
- 还行
- 不错
- 很好
- 感觉不错的样子
- 那挺好的

