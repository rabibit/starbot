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
- 恩打扰了拜拜


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
- 对对对
- 哦好好好谢谢
- 啊好，谢谢
- 哦好的好
- 恩好的谢谢啊
- 好再见
- 好我知道了


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
- 哦好好好谢谢
- 对的
- 没错
- 没问题
- 是的
- 对
- 没错订了
- ok
- yes
- 好的




## intent:book_room
@room_type: 房间|房|标间|单人间|大床房|豪华大床房|总统套房
@checkin_time: 今晚|明晚|今天|后天|周末|周六|一号|二号|十一号|二十号|二号下午|后天下午
@guest_name: 张三|李四|王麻子|杨康|穆念慈|郭啸天|李萍|杨铁心|包惜弱|完颜洪烈|柯镇恶|朱聪|韩宝驹|南希仁|张阿生|全金发|韩小莹|刘女士|王先生

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
- 那就预订[一](count)间商务标间吧
- 我觉得[两](count)间标间就足够了
- 我打算预订个[豪华套房](room_type)
- 我帮我朋友预订个[豪华套房](room_type)
- 我叫[王大锤](guest_name)订[一](count)间[大床房](room_type)[今晚](checkin_time)的电话是[12345678](guest_phone_number)
- 我想订一个[房间](room_type)
- 帮我订[一](count)间[今晚](checkin_time)的[大床房](room_type)
- 我叫[如来](guest_name)想订一个[房间](room_type)
- 你这样子[明天](checkin_time)给我留[一个](count)[三人间](room_type)
- 额，我有你这边那什么苏门卡嘛，你帮我开[两个](count)[单人大床房](room_type)
- 喂，请给我订[两个](count)[标准间](room_type)
- 那就[双人间](room_type)，订[两个](count)
- 喂你好，麻烦订[一间](count)[单人房](room_type)[今晚](checkin_time)的
- 你好帮我订一间标单
- 你好我想问一下你们有没有[单人房](room_type)
- 你好先帮我订[两间](count)[标间](room_type)
- 那留一间[标准间](number)吧
- 帮我定2间房
- 给[个](count)[标单](room_type)
- 给个[标双](room_type)

## intent: is_room_available
- 你好，[标准间](room_type)有吗
- 你好，我想问下你大床房还有吗
- 还有标双吗
- 请问有单间吗
- 帮我看下[25](number)，[26](number)得[标双](room_type)怎么样

## intent: something_like
- [0908](keyword)那种有没有

## intent: info
- 可以啊，那你开[两间](count)我[晚上](checkin_time)到嘛
- 额，[礼拜天](checkout_time)走


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
- 我叫[何任可](guest_name)


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
- [晚上10点前](time)


## intent:consultation
? 您好，这里是未来酒店，请问有什么需要呢?

- 请问你们的酒店环境怎么样
- 能简单介绍一下你们的酒店吗
- 你们的特色是什么
- 我是外地人，听朋友说你们酒店很好，可以向我介绍一下吗
- 我向了解一下你们的酒店
- 麻烦介绍一下你们的酒店
- 那你介绍介绍你们的酒店有什么特色呢


## intent:ask_for_price
- 那你们的房价是什么样的呢
- 我想了解一下价格
- 价格怎样
- 房间价格如何
- 能说说房间价格么
- 那房间价格怎么样
- 怎么住，房费多少
- 多少钱
- 什么价位
- 多少钱一包
- 小哥我想问下[套房](room_type)是多少钱一间
- 你好我想问下[双人房](room_type)是多少钱一晚的
- 我说，问一下就今天在你们酒店订一个[标间](room_type)多少钱？
- 就是这样，是那个，你们那个[大床房](room_type)跟那个[标间](room_type)价格是一样的？

## intent:is_breakfast_included
- 含早餐吗


## intent:room_available
- 还有空房间吗
- 我明天入住，还有房间吗
- 有标间吗
- 有大床房吗
- 有商务套房吗
- 有豪华套房吗
- 你那边还有双人房么


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

## intent:what_can_you_do
- 你能做什么
- 你能帮我做什么
- 你有什么功能

## intent:ask_for_help
@action: 倒茶 打水 办事儿 写作业 打卡
- 能不能帮我[倒水](action)
- 可以帮我[倒水](action)吗
- 我想你帮我[倒水](action)
- 你能帮我[倒水](action)吗
- 请问你这儿能[倒水](action)吗

## intent:other
- 中国在全面建设社会主义的进程中
- 取得了巨大的成就
- 初步奠定了现代化建设的物质文化基础
- 但也伴随着一些失误
- 1958年全国各条战线掀起了大跃进的高潮
- 同年8月
- 中共中央政治局北戴河会议
- 确定了一工农业生产的高指标
- 农业上
- 由于中共中央主席毛泽东错误地认为
- 合作社规模越大
- 越能发展生产力
- 公社化也是加速建设社会主义
- 并向共产主义过渡的最好形式
- 所以将原有的农业合作社改组成2.6万多个人民公社
- 99%的农民加入到组织中来
- 原有正常的经济体系被破坏
- 农业产值大幅度减少
- 其特点是一大二公一平二调
- 1960年开始的三年经济困难更为国民经济雪上加霜
- 1960年冬中共中央开始纠正农村工作中的“左”倾错误
- 并且决定对国民经济实行“调整、巩固、充实、提高”的方针
- 随即在刘少奇、周恩来、陈云、邓小平等的主持下
- 制定和执行了一系列正确的政策和果断的措施
- 这是这个历史阶段中的重要转变
- 1962年1月召开的有七千人参加的扩大的中央工作会议
- 初步总结了“大跃进”中的经验教训
- 开展了批评和自我批评
- 1966年到1976年，发生了给党和国家带来严重灾难的“文化大革命”
- 在瑞士的诸多种特色火锅中
- 其奶锅先是将酪搬进锅里
- 待其煮成液体状后再加入一定数量的白酒和果酒
- 吃的时候要用长柄的叉子将一块法式的面包叉起来
- 放进锅中拿出来吃
- 这时的面包又热又香
- 吃起来特别的爽口宜人
- 就这样一边烧一边蘸一边吃
- 直到火锅中的液体奶酪快要烧干烧焦时为止
- 一些嗜食瑞士奶酪火锅成性的欧洲人
- 一次甚至可以吃上二三十块蘸有液体奶酪珠面包
- 这是一种很受瑞士女孩子们特别青睐的特色火锅
- 它的食用方法和奶酪火锅差不多
- 事先将巧克力放入锅中煮成汁
- 再用长柄叉子叉着水果片
- 蘸着锅中的巧克力汁一片一片地吃
- 一直到火锅中巧克力汁蘸完为止
- 因为这种特色火锅在吃起来的时侯别具一番情趣
- 因而其在瑞士也颇受青年恋人们的喜爱
- 该国特色火锅的主要原料是牛肉片、火腿、猪排肉和虾仁等
- 配料有菠菜、洋葱以及黄油等
- 人们在吃火锅时
- 先将火锅烧热
- 然后再将菠菜和洋葱放入锅内煮一下
- 稍后再放火腿
- 鸡片和猪排肉等
- 待开始吃的时候再放入虾仁等海鲜产品
- 以保持火锅的鲜香味

## intent: ask_for_wifi_password
- 喂，那个无线网密码是什么？
- 请问wifi密码是多少
- 喂，你好，我想请问下那个，wifi是，wifi密码是多少
- 喂，我想问一下你们这边的WiFi密码是多少
- 嘿你好我想问下WiFi密码是多少
- 你好请问房间WiFi密码是多少啊
- 你好请问下wifi密码是多少
- 我是[1508](number)得客人你们房间wifi是多少
- 我想问下wifi密码

## intent: ask_for_wifi_info
- 那我们的房间的那个wifi，应该是怎么找的
- 喂，无线网是哪一个？我怎么连不上？
- 你好WiFi怎么连接

## intent: where_is_the_wenxiang
- 喂，怎么找不到哪个插电的蚊香呢？

## intent: is_there_any_wenxiang
- 有蚊香么, 有蚊子

@goods_name: 烟 泡面 饼干 水
## intent: order_something
- [茶杯](thing)送多[一个](count)[茶杯](thing)上来
- 喂送[2个](count)[杯](thing)上来
- 帮我送[2张](count)[凳子](thing)和多[几瓶](count)[水](thing)过来
- 帮我拿[四瓶](count)[矿泉水](thing)到[1802](number)
- [1802](number)送[一副](count)[扑克牌](thing)
- [1413a](number)[一副](count)[扑克](thing)钱已将给前台了
- [911](number)要[一包](count)[玉溪](goods_name)，[挂房费](pay_method)
- [饼干](goods_name)帮我拿[几个](count)上来
- 你给我送[几个](count)[酒碗](goods_name)吧
- 那给我送[一个](count)[剃须刀](goods_name)上来
- [2102](number)给我拿[两瓶](count)[矿泉水](thing)上来
- 我是[1118](number)我想问一下你们这里有[熨斗](thing)
- 哪个[1518](number)给拿[两幅](count)[扑克牌](thing)
- 喂你好，我这边是[1718](number)房，给我送[5,6个](count)[衣架](thing)把，我要洗一下衣服
- 你好[1906](number)有[蓝龙](thing)么，送[一包](count)
- 你好我想问一下有没有[剪刀](thing)可以借一下么
- 你好，请问可以送[两个三个](count)[衣架](thing)给我好么716
- 你好还有[凳子](thing)么，拿[几张](count)凳子上来
- 你好能给我们房间送[个](count)[袋子](thing)过来
- [911](number)要[一包](count)[玉溪](thing)[挂房费](pay_method)
- 喂，你好请帮我拿[一个](count)[充电器](thing)上[915](number)
- 点一个[小炒牛肉](thing)
- 要个[青椒炒蛋](thing)
- 还有[饭](thing)
- 你好我这里是[四条2](number)房间号，麻烦帮我们搬[几张](count)[凳子](thing)过来可以吗
- 麻烦拿[10个](count)那个[一次性纸杯](thing)到[2208](number)来
- 喂，拿[一套](count)那个[牙刷](thing)到[1906](number)
- 你好帮我送[个](count)[有靠背得椅子](thing)到[1409](number)
- [1918](number)要[2碗](count)[米饭](thing)，在给我来[一笼](count)[香煎生肉包](thing)可以吗
- [716](number)要多[2个](count)[衣架](thing)
- 我们是[1808](number)，让客房中心送[2瓶](count)[水](thing)[2双](count)[一次性拖鞋](thing)过来
- 来一份[叉烧包](thing)
- 来一个[干炒河粉](thing)
- 麻烦帮我告诉客房中心叫他帮我拿[2个](count)[牙刷](thing)过来
- 前台叫一下服务员拿一下[垃圾袋](thing)吧，就是每天都要装那个湿衣服的
- 你好我想要[衣架](thing)
- 帮我送[个](count)[快餐垫](thing)上来

## intent: ask_for_something_to_eat
- 喂你好，前台有什么东西吃的
- 没有了，哪有什么吃得

## intent: are_you_sure
- 什么东西都没有啊

## intent: is_there_xxx
- 你好[吹风机](thing)有没有
- 喂，你好，那个[吹风机](thing)有没有啊？
- 喂，你们那边有没有[卷尺](thing)啊
- 我想问一下你们这里还有[宵夜](thing)没有啊？
- 你好我想问下有[指甲剪](thing)吗
- [泡面](thing)也没有啊？
- 你们酒店没有[剃须刀](thing)的么
- 那个我问下 ，你们前台有没有[指甲剪](thing)
- 你好，有没有[茶具](thing)
- 那等一下，有[椰树椰奶](thing)么
- 唉，你好前台有[方便面](thing)吗
- 你们酒店没有[剃须刀](thing)吗
- 你好你这有[烟](thing)吗
- 你们酒店有[剃须刀](thing)吗
- 上面有[冰箱](thing)吗
- 你好我想问下你们哪儿是有[卫生纸](thing)吗
- 有[炒粉](thing)吗
- 你们酒店有[按摩](thing)得吗
- 房间不是有[牙刷](thing)吗

## intent: ask_for_awaking
- 你好，[明天早上7点10分](time)叫我起床好吧
- 唉，前台，[1917](number)[明天7点10分](time)叫醒我行吗

## intent: ask_for_something_to_drink
- 饮料有没有啊
- 你好你们前台有饮料有

## intent: number_of_thing
- 那拿[两瓶](count)上[1808](number)号房

## intent: 
- 然后叫客房中心那个[一头](count)[肥皂](thing)上来

## intent: info
- [挂房费](pay_method)
- [12345张](count)

## intent: hmm
- 嗯嗯
- 额
- 诶恩

## intent: breakfast_ticket_not_found
- 我这边是[1107](number)，怎么打扫卫生没有留明天早餐票给我呢
- 我是1807的，你昨天晚上开房没有给早餐票吗
- 我想问下你们今天问早餐券

## intent: is_there_breakfast
- 这边提供早餐么
- 你好我想问下你们这个有没有早餐送的
- 这里是不是有早餐吃的啊？

## intent: where_to_have_breakfast
- 你好，请问早餐在几楼?
- 你们早点在几楼啊
- 问一下这个早餐在几楼
- 你好请问早餐在几楼啊

## intent: when_to_have_breakfast
- 我想问一下那个早餐是几点到几点
- 你们这早餐到多少

## intent: when_to_have_lunch
- 14号中午，想问一下这些中餐的话都是什么时候上齐

## intent: is_there_breakfast_now
- 还有早餐吃么

## intent: is_it_over
- 现在结束了没有啊？
- 现在结束没啊
- 那现在不是结束了

## intent: other_issue_needs_service
- 你好[1012](number)需要客房服务，[马桶堵了](topic)
- 喂你好我是[1209](number)的住户，能帮我们[关一下窗](topic)么
- 我觉得那个[洗手间的灯很暗](topic)
- 这个[床单好像有血](topic)你上来看看
- 我这边是[511](number)房间，我进来[有一股很奇怪的味道](topic)，你们能叫人过来处理一下么
- 你好我是[518](number)的 你们[下面KTV有点吵](topic)
- 你好麻烦叫个人过来给我[搞一下电脑](topic)
- 恩我这边是出现一个[升级异常](topic)

## intent: tv_problem
- 那好我是[1208](number)的房，那个电视怎么开
- 你喊一下[2109](number)号房来帮我们开一下电视机
- [1403](number)电视开不了
- 我是[2113](number)我的电视没有图像
- 那个电视开不了了，怎么样都开不了电视
- 恩我[1108](number)这里电视放不了
- [803](number)这个电视怎么开不了呢
- [1710](number)电视有点问题，怎么开不了啊
- 我这里是[1913](number)，电视怎么没有信号，打开没有信号
- 你好麻烦帮看下[2230](number)的电视，老打不开
- 帮我找个人看一下电视都放不出来
- 你好这个电视机怎么自己又灭了
- 这边是[1008](number)，我和他们换了，这个遥控不知道怎么开电视
- 你好这边[603](number)，电视看不了，找个人来看一下
- 你好我电视放不了
- 这个电视怎么看不了啊
- 这个电视没有信号得，按那个键
- 你好，麻烦叫服务员帮我开下电视好吗，我这开不开啊

## intent: ask_to_change_room
- 你好，[715](number)就是异味很重哦，可以换间房吗，首房啊
- 你好，我这边是[1515](number)刚入住的，我想一下我可以换其他房间么
- 你好，[151](number)我这么久都没人上来你给我换别的房间可以吗
- 那你给我换个[双标](room_type)

## intent: ask_price_for_changing_room
- 你好那个换房是加[20](number)块是吗

## intent: ask_for_food
- 这边还有[快餐](food_name)么?
- 你好上面有[粉](food_name)吃么，现在

## intent: not_found
- 好像找过没有诶

## intent: chitchat
- 那我怎么办，我指甲剪给撇掉了

## intent: leave_over_something
- 你好我是昨天住在你们1713的客人，退房的时候我有[一副眼镜](thing)忘记拿了
- 你帮我问一下看看那个清洁阿姨有没有见到，[1713](number)

## intent: info
- 额，你好，额。。我，我柳州的，叫马克帧那，经常在你这里住的呀

## intent: stay_extension
- 你好我想问一下，晚上我可以房间继续
- 你好帮我看一下[2016](number)续住么
- 我想问下晚上我们这个房间可不可以续住

## intent: is_it_free
- 你好我想问一下你，你们的[矿泉水](thing)和[茶叶](thing)是免费的么？

## intent: complain
# TODO: 训练情感分析
- 恩什么都弄不了
- 六点钟就开始了，闷得要死，我睡都睡不了
- 那个之前问的网线怎么还没有送上来
- 到底能不能送，你直接给个话啊

## intent: ask_to_clean_room
- 那个把[308](number)打扫一下啊
- 我房间需要打扫我是[1109](number)
- 现在可以来这边打扫一下房间吗
- 麻烦你叫一下服务员来，刚才她来过一遍，因为我们有人正在洗澡，所以没有清理那个垃圾，可不可以现在叫服务员过来帮我们清理下垃圾
- 可不可以现在叫服务员过来帮我们清理下垃圾
- 麻烦你好帮我打扫一下卫生
- 我这是[1806](number)号房，我这里有套被子很脏可以叫人帮我换一下吗

## intent: air_conditioner_problem
- 你好，我想问一下那个空调那个一直显示off那个怎么打开
- [1310](number)空调不行都没有冷气的
- 我想问下你们的空调是不是调了，是调了温度还是什么不怎么制冷了
- 你好我是[913](number)的房子，这个空调漏水哦
- 你好我想问一下，这个空调我不太会用，是不是坏了
- 你好我想问哈，空调一直显示off怎么打开
- 房间内空调打不开了好像一直显示off
- 那个空调我不懂开
- 那个中央空调怎么开不了
- 我是[2211](number),空调没有变冷，每天晚上都是热得

## intent: network_problem
- 为什么没有网络呢

## intent: ask_for_dry_tea
- 我的房间里都好几天没有茶叶了

## intent: ask_to_open_door
- 诶，请叫服务员给我开下房门下，[1813](number)
- 麻烦你帮我叫人开一下[1918](number)的房嘛
- 我[1307](number)的，哪个房卡锁在里面了，能帮我开一下门么
- 你好麻烦叫服务员给我开一下[1302](number) 我的卡开不了
- 麻烦那你叫人帮我开下[1918](number)得房嘛
- 麻烦开下[5楼](floor)得门好吗
- 你好我是离子监狱得，我们有个同事房卡掉了能帮我们开下房吗

## intent: ask_to_change_thing
- 我是1507的客人，我感觉那个[床单](thing)很湿很潮，可不可以给我换一个
- 把两个床的[床单](thing)换下啊，好久没换了吧

## intent: ask_for_phone_number
- [餐厅](subject_of_phone_number)电话是多少
- [楼上KTV](subject_of_phone_number)电话号码是多少啊

## intent: ask_for_traffic_info
- 我想问一下，去南宁的汽车

## intent: where_is_tv_controller
- 请问房间里是不是应该还有个电视机的遥控器，我只看到一个机顶盒的遥控器

## intent: info
- 你好我美团上面订了一间房
- [十二点](checkin_time)，[一点](checkin_time)前来可以么，但是我已经给钱了

## intent: delay_checkin
- 恩大概要晚一点

## intent: ask_for_charger
- 你好你们这边充电器么
- 你好我是[613](number)，我忘记带充电器了，房间或者你们有么
- 你好，你这里有[苹果](charger_type)充电器吗
- 你好，我们是住在[1109](number)得然后这里有没有充电线

## intent: is_my_room_ready
- 你好我想问一下那个房间可以用了么

## intent: is_my_cloth_ready
- 我想问一下我的衣服干没有
- 那个[601](number)那个衣服洗好了吗，烘干了吗
- 我问下帮我晒得衣服干了没有

## intent: ask_for_laundry
- 洗衣物在哪里
- 你好可以帮洗衣服么
- 你好，我们想了解洗衣房的预定
- 你好还在洗衣服么
- 你好，我想问下你们有烘干衣服的服务么
- 你好我想洗衣服
- 喂你好，我想问下[现在](time)可以上去洗衣服吗
- 好的[现在](time)可以拿上去洗是吧
- 你好可以洗衣服吗[现在](time)
- [现在](time)洗衣房可以洗衣服吗

## intent: is_there_cloth_drier
- 那个烘干机有空得可以用吗
- 啊姐，上面可以晒衣服吗
- 你好我想问下你们酒店是有烘干衣服的吗
- 我想问下外面有露台那个地方可以晒衣服吗，我们打球衣服晒不干

## intent: cloth_not_dry
- 你[802](number)有个衣服这里洗了，不够干，我这里没有衣架了，想送过去还给你

## intent: how_to_call
- 你好我请问一下我们房间的电话怎么打
- 你好我想问下，你们这个电话打外地手机是怎么打得
- 请问下打外线怎么打

## intent: is_there_xxx_around
- 恩，附近有[超市](thing)吗
- 咱们旁边有个[桑拿](thing)是吧？

## intent: how_far_is_the_nearest_store
- 附近超市有多远

## intent: conplain
- 这么[远](distance)

## intent: give_up
- 那没办法了谢谢

## intent: comfirm_location
- 是[南宁](location)吗

## intent: cancel_book_room
- 不好意思我定错了我可以取消吗，我在南宁
- 我在南宁不好意思定错了，我没看
- 你先看下能取消吗

## intent: charger_type
- [安卓](charger_type)的

## intent: can_deliver?
- 可以送餐吗

## intent: any_other?
- 还有什么菜
- 除了套房以外得比较好的房间是什么

## intent: query_checkout_time
- 我想问一下最迟几点退房
- 总机我想问一下我们这个退房是在两点以前吧

## intent: is_vip_the_same
- 会员也是这样吗

## intent: re_confirm
- 最迟一点半是吧

## intent: buy_or_borrow
- 要[一个](count)是买吗

## intent: info
- 那好，我是[813a](number)
- [现在](checkin_time)过去

## intent: this_phone
- 肯定啊就这个号码

## intent: how_can_i_do
- 那我怎么办

## intent: query_book_record
- 能帮我看下[唐莲](guest_name)的预订有吗

## intent: ask_if_ferry
- 你好我刚刚在美团定了一间房，我想问问你们那是不是有车接送得

## intent: how_much_if_stay_until
- 我[四点钟](time)退房收多少钱啊

## intent: urge
- 你好你这个网络什么时候能好啊

## intent: is_it_ok
- 那个我身份证丢了，我在车站开得身份证名可以用吗

## intent: query_agreement_price
- 问下[税务局](org)协议价多少钱

## intent: fetch_it_myself?
- 喔等下直接下去拿吗

## intent: can_i_have_invoice
- 我问下你们那是可以开发票得吧

## intent: wanna_more
- 我这边是想多开可以吗
- 不够

## is_meal_available_now
- 你好现在可以吃饭了吗

## intent: ask_for_more_breakfast_ticket
- 你好我们那个只有一张早餐票，我们两个人
- 你好我们开那个房的得一张早餐票的我们开双人房的
- 915只给了我一张早餐卷

## intent: query_dinner_time
- 我想点一下餐最晚到几点
- 我想问下晚餐是多久开始

## intent: query_foods
- 哪里有什么[吃得](keyword)
- [包子](keyword)有什么

## intent: and_xxx
- 恩还有[水](thing)啊

## intent: lack_of_thing
- 但[牙刷](thing)只有一根，不够用啊

## intent: how_much_did_i_spend
- 帮我看哈我昨天在03号厢消费了多少

## intent: is_manager_there
- 喂，请问你们酒店的经理在吗？

## intent: info
- 我是那个一旅的导游，那个今天晚上有人在那里住一下

## intent: meituan_ticket_comfirm
- 你好，这边是美团的，订单尾号[3320](number)[胡冬丽](guest_name)[今天](checkin_time)入住[一晚](duration)[商务单人房](room_type)[一间](count)，[198](price)含双早订单可以接吗？

## intent: is_breakfast_custom?
- 你好，我想问一下你们早餐是那种自助餐吗？

## intent: i_have_booked_some_room
- 喂，你好，刚刚我在携程上订了你们那个[5个](count)[房间](room_type)，[4个](count)[大床房](room_type)，然后[1个](count)[双床房](room_type)
