#!/usr/bin/env python3
"""Seeded SafeBelt synthetic data. External dependencies: pandas and numpy only.
All names/accounts are synthetic. Times are local Asia/Bangkok without an offset.
Run: python generate_safebelt.py --users 2500 --seed 20261013 --out data_v7
"""
import argparse
import collections
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

END = pd.Timestamp('2026-09-30 23:59:59')
EARLIEST = pd.Timestamp('2024-09-30')
LATEST = pd.Timestamp('2026-08-30')
TXCOLS = 'txn_id customer_id account_id posted_datetime amount direction txn_type channel balance_after description counterparty_account_id counterparty_bank_code counterparty_name promptpay_proxy_type biller_id ref1 ref2 mandate_id mcc merchant_id merchant_name original_currency original_amount'.split()
LBCOLS = 'txn_id gt_kind gt_series_id gt_flags gt_link_txn_id'.split()
CUCOLS = 'customer_id customer_name first_account_open_date'.split()
ACCOLS = 'customer_id account_id account_type open_date'.split()
SRC = [
 'https://www.bot.or.th/content/dam/bot/content-fragments/holiday-calendar/en/bot-notification-holiday/BOT-Notification-Holiday-Calendar-2024-EN.pdf',
 'https://www.bot.or.th/en/news-and-media/news/news-20231102.html',
 'https://www.bot.or.th/en/news-and-media/news/news-20240221.html',
 'https://www.bot.or.th/content/dam/bot/content-fragments/holiday-calendar/en/bot-notification-holiday/BOT-Notification-Holiday-Calendar-2025-EN.pdf',
 'https://www.bot.or.th/en/news-and-media/news/news-20241127.html',
 'https://www.bot.or.th/content/dam/bot/fipcs/documents/FPG/2568/EngPDF/25680162.pdf']
# Nationwide commercial-bank holidays; regional/Islamic-bank additions excluded.
# 2024 Jan 2 was removed by the later BOT announcement.
HOLIDAYS = set('''2024-01-01 2024-02-26 2024-04-08 2024-04-12 2024-04-15 2024-04-16 2024-05-01 2024-05-06 2024-05-22 2024-06-03 2024-07-22 2024-07-29 2024-08-12 2024-10-14 2024-10-23 2024-12-05 2024-12-10 2024-12-31
2025-01-01 2025-02-12 2025-04-07 2025-04-14 2025-04-15 2025-05-01 2025-05-05 2025-05-12 2025-06-02 2025-06-03 2025-07-10 2025-07-28 2025-08-11 2025-08-12 2025-10-13 2025-10-23 2025-12-05 2025-12-10 2025-12-31
2026-01-01 2026-01-02 2026-03-03 2026-04-06 2026-04-13 2026-04-14 2026-04-15 2026-05-01 2026-05-04 2026-06-01 2026-06-03 2026-07-28 2026-07-29 2026-08-12 2026-10-13 2026-10-23 2026-12-07 2026-12-10 2026-12-31'''.split())

def business(d, step=-1):
    while d.weekday() >= 5 or str(d.date()) in HOLIDAYS:
        d += pd.Timedelta(days=step)
    return d

def paydays(start, rule):
    dates = []
    for m in pd.date_range(start.replace(day=1), END.normalize(), freq='MS'):
        if rule == 'semimonthly':
            dates.extend([business(m.replace(day=15)), business(m + pd.offsets.MonthEnd(0))])
        elif rule == 'last_business_day':
            dates.append(business(m + pd.offsets.MonthEnd(0)))
        elif rule == 'first_business_day':
            dates.append(business(m, 1))
        elif rule == 'day28':
            dates.append(m.replace(day=28))
        else:
            dates.append(business(m.replace(day=25)))
    return [d for d in dates if start <= d <= END]

def monthly(start, day, months=1):
    ds = [m.replace(day=min(day, m.days_in_month)) for m in pd.date_range(start.replace(day=1), END.normalize(), freq='MS')]
    return [d for d in ds if d >= start][::months]

def cents(value):
    return max(1, int(round(float(value)*100)))

def merchant_code(name):
    return 'M'+''.join(c for c in name.upper() if c.isalnum())

class Customer:
    def __init__(self, i, persona, second, heldout, seed):
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        self.i, self.cid = i, f'C{i:05d}'
        self.name = f'{self.pick(["นาย", "น.ส."])} {self.pick(["พิมพ์", "สมชาย", "มาลี", "นภา", "ธนา", "กานต์", "วริน", "ปกรณ์", "อร", "ศศิ"])} {self.pick(["ก", "ข", "จ", "ว", "น"])}****'
        self.persona = persona
        # The 2026 baseline floor leaves room for earlier April raises and job
        # changes while keeping realized historical monthly salary >= THB 15k.
        self.salary = round(float(np.clip(self.rng.lognormal(np.log(28000), .43), 18000, 100000))/100)*100
        self.raise_rate = float(self.rng.uniform(.03, .05))
        maximum = pd.Timestamp('2026-04-01') if persona != 'freelancer' else LATEST
        self.open = EARLIEST + pd.Timedelta(days=self.integer(0, (maximum-EARLIEST).days))
        self.main = f'A{i:05d}01'
        self.second = f'A{i:05d}02' if second else ''
        self.second_open = min(self.open + pd.Timedelta(days=self.integer(0, 45)), LATEST) if second else None
        self.account_open = {self.main:self.open}
        if second: self.account_open[self.second] = self.second_open
        self.opening = {a:cents(self.salary*self.rng.uniform(.5, 2)) for a in self.account_open}
        self.external_opening = cents(self.salary*self.rng.uniform(.5, 2))
        self.style = self.pick(['frugal','foodie','shopper','commuter','cafe_lover','traveller','family_supporter','heavy_spender'])
        self.housing = self.pick(['rent','rent','dorm','with_family'])
        self.split = 'heldout' if heldout else 'train'
        self.events, self.series, self.smap = [], [], {}
        self.etoken = 0
        self.salary_dates = []
        self.skipped = collections.Counter()
        self.rule = 'irregular'
        self.new_rule = ''
        self.job_change = None

    def integer(self, a, b): return int(self.rng.integers(a, b+1))
    def chance(self, p): return bool(self.rng.random() < p)
    def pick(self, xs): return xs[self.integer(0, len(xs)-1)]
    def amount(self, base, cv=.08): return max(1, round(float(base*self.rng.lognormal(-cv*cv/2, cv)), 2))
    def time(self, d, payroll=False):
        return d.normalize() + pd.Timedelta(seconds=self.integer(60, 6*3600-1) if payroll else self.integer(7*3600, 23*3600-1))

    def series_add(self, name, kind, category, cycle='calendar', period=30.4, essential=False, status='active', start=None, stop=None, quirks=''):
        sid = f'S{self.i:05d}{len(self.series)+1:03d}'
        row = dict(series_id=sid, customer_id=self.cid, name=name, kind=kind, category=category, status=status, cycle_type=cycle, true_period_days=period, essential=essential, quirks=quirks, start_date=str((start or self.open).date()), end_date=str(stop.date()) if stop is not None else '', observed_transactions=0, successor_series_id='')
        self.series.append(row); self.smap[sid]=row
        return sid

    def quirk(self, sid, q):
        if not sid: return
        old = self.smap[sid]['quirks'].split(';') if self.smap[sid]['quirks'] else []
        if q not in old: self.smap[sid]['quirks']=';'.join(old+[q])

    def event(self, d, amount, kind, sid='', flags='', link='', account=None, **fields):
        account = account or self.main
        if d < self.account_open[account] or d > END: return None
        self.etoken += 1
        row = {k:'' for k in TXCOLS}
        row.update(customer_id=self.cid, account_id=account, posted_datetime=d, amount=cents(amount), direction='DR', txn_type='qr_payment', channel='KPLUS')
        row.update(fields)
        row.update(_kind=kind,_sid=sid,_flags=flags,_link=link,_token=f'E{self.etoken:07d}',_own_dest='')
        self.events.append(row)
        return row

    def cp(self, key, name, bank='014', proxy=''):
        return dict(counterparty_account_id=f'X{self.i:05d}{key}', counterparty_bank_code=bank, counterparty_name=name, promptpay_proxy_type=proxy)

    def income(self):
        if self.persona == 'freelancer':
            sid=self.series_add('freelance client receipts','income','freelance','irregular',17,True)
            d=self.open + pd.Timedelta(days=self.integer(0, 7))
            while d<=END:
                gap=self.integer(10,24)
                self.event(self.time(d),self.amount(self.salary*gap/30.4,.25),'income',sid,direction='CR',txn_type='transfer_in',channel='KPLUS',description='โอนจาก ลูกค้า งานบริการ',**self.cp('CLIENT01','บจก. ลูกค้าสมมติ'))
                d+=pd.Timedelta(days=gap)
            self.salary_dates=monthly(self.open, 25)
            return
        self.rule='semimonthly' if self.persona=='salaried_semimonthly' else self.pick(['day25','last_business_day','day28','first_business_day'])
        if self.persona=='salaried' and self.open<pd.Timestamp('2025-09-01') and self.chance(.09):
            self.job_change=pd.Timestamp('2026-02-01')
            self.new_rule=self.pick([r for r in ['day25','last_business_day','day28','first_business_day'] if r!=self.rule])
        segments=[(self.open,self.job_change,self.rule,1)]
        if self.job_change: segments.append((self.job_change,None,self.new_rule,2))
        bonus=self.chance(.32)
        for start, stop, rule, emp in segments:
            employer=f'{self.pick(["SIAM", "BANGKOK", "THAI", "NOVA"])} {self.pick(["TECH", "RETAIL", "SERVICES", "FOODS"])} {self.i:05d}-{emp}'
            sid=self.series_add('salary '+employer,'income','salary','calendar',15.2 if rule=='semimonthly' else 30.4,True,'switched' if stop is not None else 'active',start,stop, 'annual_april_raise')
            ds=[d for d in paydays(start,rule) if stop is None or d<stop]
            self.salary_dates.extend(ds)
            for d in ds:
                raises_remaining=sum(d<pd.Timestamp(f'{y}-04-01') for y in [2025,2026])
                monthly_salary=self.salary/(1+self.raise_rate)**raises_remaining
                if self.job_change and d<self.job_change: monthly_salary/=1.08
                self.event(self.time(d,True),round(monthly_salary/(2 if rule=='semimonthly' else 1),2),'income',sid,direction='CR',txn_type='payroll_in',channel='AUTO',description='PAYROLL '+employer,**self.cp(f'EMP{emp:02d}',employer,'004'))
            if bonus:
                for year in [2024,2025,2026]:
                    d=pd.Timestamp(year=year,month=12 if self.i%2 else 4,day=10)
                    if start<=d<=END and (stop is None or d<stop):
                        self.event(self.time(d,True),self.salary*self.rng.uniform(.3,1),'income',sid,'bonus',direction='CR',txn_type='payroll_in',channel='AUTO',description='PAYROLL '+employer,**self.cp(f'EMP{emp:02d}',employer,'004'))
        self.salary_dates.sort()

    def payment(self, date, amount, sid, mode='bill', merchant='SERVICE', mcc=None, refkey='01', flags='', link='', cp=None, usd=False):
        sr=self.smap[sid]; kind=sr['kind']
        if mcc is None:
            mcc={'SHOPEE':'5311','NETFLIX.COM':'4899','SPOTIFY':'4899','YOUTUBE':'4899','DISNEY+':'4899','APPLE.COM/BILL':'5734','GOOGLE ONE':'5734','CLOUDPLUS.COM':'5734'}.get(merchant,'5999')
        if mode=='p2p':
            return self.event(date,amount,kind,sid,flags,link,txn_type='promptpay_out',channel='KPLUS',description='PromptPay To: '+cp['counterparty_name'],**cp)
        if mode in ['card','qr','counter']:
            if mode=='counter': merchant,mcc='7-ELEVEN','5499'
            org=round(amount/34.1,2) if usd else amount
            if usd: amount=round(org*float(self.rng.uniform(33.0,36.5)),2)
            return self.event(date,amount,kind,sid,flags,link,txn_type='card_purchase' if mode=='card' else 'qr_payment',channel='CARD' if mode=='card' else 'KPLUS',description=self.pick([merchant,merchant+' TH',merchant+' BANGKOK']),merchant_id=merchant_code(merchant),merchant_name=merchant,mcc=mcc,original_currency='USD' if usd else 'THB',original_amount=org)
        fields=dict(txn_type='direct_debit' if mode=='auto' else 'bill_payment',channel='AUTO' if mode=='auto' else 'KPLUS',description=merchant+' REF'+refkey,biller_id='BL'+merchant.replace(' ','').replace('.',''),ref1='00'+f'{self.i:05d}'+refkey,ref2=date.strftime('%Y%m') if sr['category'] in ['utilities','telecom'] else '',mandate_id='MD'+sid if mode=='auto' else '')
        return self.event(date,amount,kind,sid,flags,link,**fields)

    def calendar_bill(self,name,category,amount,day,mode='bill',merchant='SERVICE',period_months=1,essential=True,cv=0,stop=None,status='active',start=None,cp=None,hard=False,usd=False,pause=False):
        start=start or self.open
        sid=self.series_add(name,'obligation',category,'calendar',{1:30.4,3:91.3,6:182.6,12:365.25}[period_months],essential,status,start,stop)
        ds=monthly(start,day,period_months)
        if stop is not None: ds=[d for d in ds if d<stop]
        late=hard and self.chance(.18)
        catch=self.integer(1,len(ds)-2) if hard and len(ds)>4 and self.chance(.10) else -10
        changed=cp is not None and len(ds)>6 and self.chance(.18)
        change_date=ds[len(ds)//2] if changed else END+pd.Timedelta(days=1)
        paused_indices=set([len(ds)//2,len(ds)//2+1]) if pause and len(ds)>5 else set()
        if paused_indices: self.smap[sid]['status']='paused_resumed';self.quirk(sid,'paused_two_cycles')
        for j,due in enumerate(ds):
            billing_period=due.strftime('%Y%m')
            if j==catch: self.quirk(sid,'skipped_then_catch_up');continue
            if j in paused_indices:continue
            flags=[]; amt=self.amount(amount,cv) if cv else round(amount,2)
            if category=='utilities' and 'electric' in name:
                amt=round(amt*(1.28 if due.month in [3,4,5,6] else .88),2)
            if category in ['streaming','cloud'] and due>=pd.Timestamp('2026-05-01'):
                amt=round(amt*1.12,2);self.quirk(sid,'price_increase')
            cpx=dict(cp) if cp else None
            if changed and due>=change_date:
                cpx['counterparty_account_id']+='NEW';cpx['counterparty_bank_code']='002'
                due=due.replace(day=min(day+4,28));self.quirk(sid,'payee_account_changed');self.quirk(sid,'due_day_changed')
            if j==catch+1:amt=round(amt*2,2);flags.append('catch_up_double')
            if late and self.chance(.35):due+=pd.Timedelta(days=self.integer(4,15));flags.append('paid_late');self.quirk(sid,'late_payer')
            paymode=mode
            if hard and mode=='bill' and self.chance(.07):paymode='counter';flags.append('paid_at_counter');self.quirk(sid,'counter_payment')
            if mode=='card' and self.chance(.10):due+=pd.Timedelta(days=self.integer(1,2));flags.append('posting_delay');self.quirk(sid,'posting_delay')
            d=self.time(due)
            if cpx and hard and self.chance(.06) and d+pd.Timedelta(days=2)<=END:
                part=round(amt*.6,2)
                a=self.payment(d,part,sid,mode,merchant,refkey=sid[-3:],flags='|'.join(flags+['split_part1']),cp=cpx)
                if a:self.payment(d+pd.Timedelta(days=2),round(amt-part,2),sid,mode,merchant,refkey=sid[-3:],flags='split_part2',link=a['_token'],cp=cpx)
                self.quirk(sid,'split_payment')
                continue
            a=self.payment(d,amt,sid,paymode,merchant,refkey=sid[-3:],flags='|'.join(flags),cp=cpx,usd=usd)
            if a and a['ref2']:a['ref2']=billing_period
            if a and mode=='card' and self.chance(.035) and d+pd.Timedelta(days=7)<=END:
                actual=a['amount']/100
                dup=self.event(d+pd.Timedelta(minutes=2),actual,self.smap[sid]['kind'],sid,'duplicate_charge',a['_token'],txn_type='card_purchase',channel='CARD',description=a['description'],merchant_id=a['merchant_id'],merchant_name=a['merchant_name'],mcc=a['mcc'],original_currency=a['original_currency'],original_amount=a['original_amount'])
                if dup:
                    self.event(d+pd.Timedelta(days=self.integer(2,7)),actual,'refund',sid,'duplicate_refund',dup['_token'],direction='CR',txn_type='refund_in',channel='CARD',description='REFUND '+merchant,merchant_id=a['merchant_id'],merchant_name=merchant,mcc=a['mcc'],original_currency=a['original_currency'],original_amount=a['original_amount'])
                    self.quirk(sid,'duplicate_and_refund')
        return sid

    def bills(self):
        salary=self.salary
        landlord=self.cp('RENT01','นาย สมชาย ก****','014','mobile')
        if self.housing!='with_family':
            rent=round(min(22000,salary*self.rng.uniform(.16,.25))/100)*100
            self.calendar_bill('rent','p2p',rent,self.pick([1,1,5]),'p2p',cp=landlord,hard=True)
            if self.housing=='dorm':self.calendar_bill('dorm utilities','utilities',self.rng.uniform(350,800),8,'p2p',cp=landlord,cv=.20,hard=True)
        self.calendar_bill('electricity','utilities',min(1700,salary*.028),self.integer(9,22),'bill',self.pick(['MEA ELECTRICITY','PEA ELECTRICITY']),cv=.18,hard=True)
        self.calendar_bill('water','utilities',self.rng.uniform(80,240),self.integer(7,23),'bill','MWA WATER',cv=.20)
        self.calendar_bill('mobile postpaid','telecom',self.pick([299,399,499,699]),self.integer(5,25),'auto',self.pick(['AIS MOBILE','TRUE MOBILE','DTAC MOBILE']))
        if self.chance(.15):self.calendar_bill('mobile second line','telecom',299,18,'bill','AIS MOBILE')
        if self.chance(.55):self.calendar_bill('home internet','telecom',self.pick([399,499,599]),self.integer(7,25),'auto','HOME FIBRE')
        if self.chance(.43):self.calendar_bill('credit card statement','credit_card',salary*.055,12,'bill','CARD ISSUER',cv=.40,hard=True)
        if self.chance(.25):self.calendar_bill('personal loan','installment',salary*.055,7,'auto','PERSONAL FINANCE')
        if self.chance(.38):
            start=self.open+pd.Timedelta(days=self.integer(0,25));stop=start+pd.DateOffset(months=self.integer(6,12))
            self.calendar_bill('Shopee PayLater installment','installment',self.pick([499,699,899,1299]),10,'card','SHOPEE',start=start,stop=stop,status='completed' if stop<=END else 'active')
        if self.chance(.20):self.calendar_bill('gym membership','fitness',self.pick([499,699,999]),6,'auto','FITNESS CLUB',essential=False,pause=self.chance(.6))
        subs=[('netflix subscription','NETFLIX.COM',219),('spotify subscription','SPOTIFY',129),('youtube premium','YOUTUBE',179),('disney subscription','DISNEY+',199),('icloud storage','APPLE.COM/BILL',35),('google one storage','GOOGLE ONE',70),('USD cloud subscription','CLOUDPLUS.COM',119)]
        n=min(5,1+int(salary/22000)+self.integer(0,1))
        for idx in self.rng.choice(len(subs),size=n,replace=False):
            name,merchant,amt=subs[idx];isusd=name.startswith('USD')
            stop=None;status='active';switch=False
            if self.open<pd.Timestamp('2026-01-01') and self.chance(.16):
                stop=pd.Timestamp('2026-04-01');status=self.pick(['cancelled','switched']);switch=status=='switched'
            old=self.calendar_bill(name,'cloud' if 'storage' in name or isusd else 'streaming',amt,self.integer(3,24),'card',merchant,essential=False,stop=stop,status=status,usd=isusd)
            if switch:
                new=self.calendar_bill(name+' annual plan','streaming',amt*10,15,'card',merchant,12,False,start=stop,usd=isusd)
                self.smap[old]['successor_series_id']=new;self.quirk(old,'monthly_to_yearly')
        if self.chance(.40):self.calendar_bill('life insurance','insurance',min(18000,salary*.22),self.integer(5,24),'bill',self.pick(['AIA','MTL','FWD']),self.pick([3,6,12]),True)
        if self.chance(.20):self.calendar_bill('motor insurance','insurance',self.rng.uniform(2000,5500),17,'bill','MOTOR INSURANCE',12)
        if self.chance(.14):self.calendar_bill('condo common fee','housing',salary*.07,5,'bill','CONDO MANAGEMENT',3)
        if self.chance(.25):
            # Annual due dates specifically in the June-July student-loan window.
            d=pd.Timestamp(year=self.open.year,month=self.pick([6,7]),day=self.integer(5,24))
            if d<self.open:d+=pd.DateOffset(years=1)
            self.calendar_bill('student loan กยศ','government',salary*.13,d.day,'bill','กยศ',12,start=d)
        if self.chance(.23):self.calendar_bill('vehicle tax','government',self.rng.uniform(500,1800),20,'bill','DLT VEHICLE TAX',12)
        if self.chance(.11):
            sid=self.calendar_bill('rotating savings club แชร์','p2p',salary*.035,5,'p2p',cp=self.cp('CLUB01','น.ส. มาลี ว****','004','mobile'))
            self.smap[sid]['kind']='recurring_transfer'
            for e in self.events:
                if e['_sid']==sid:e['_kind']='recurring_transfer'

    def intervals(self):
        opts=[('drinking water delivery','utilities',self.integer(10,20),90,'obligation','WATER DELIVERY'),('pet food','pets',self.integer(18,35),390,'obligation','PET SHOP'),('medicine refill','pharmacy',self.pick([28,30,56,84]),250,'obligation','PHARMACY'),('contact lenses','pharmacy',self.integer(30,60),350,'obligation','OPTICAL SHOP'),('cooking gas','utilities',self.integer(35,60),430,'obligation','GAS DELIVERY'),('car wash','transport',self.integer(12,25),180,'habit','CAR WASH'),('laundry','household',self.integer(6,10),90,'obligation','LAUNDRY'),('badminton court','sports',self.pick([7,14]),140,'obligation','BADMINTON COURT'),('house cleaning','household',14,350,'obligation','HOUSE CLEANING')]
        for idx in self.rng.choice(len(opts),size=self.integer(2,4),replace=False):
            name,cat,period,amt,kind,merchant=opts[idx]
            sid=self.series_add(name,kind,cat,'interval',period,kind=='obligation')
            d=self.open+pd.Timedelta(days=self.integer(0,period-1))
            while d<=END:
                if self.chance(.09):self.quirk(sid,'skipped_cycles')
                else:self.payment(self.time(d),self.amount(amt,.07),sid,'qr',merchant,'5912' if cat=='pharmacy' else '5999')
                d+=pd.Timedelta(days=period)
        # Deliberately misleading patterns: repeated spending is not a bill.
        for name,cat,period,amt,merchant,cycle in [('weekly groceries','groceries',7,450,'LOTUS','interval'),('haircut','personal_care',36,250,'HAIR STUDIO','irregular'),('e-wallet topups','wallet',9,150,'WALLET','irregular')]:
            if not self.chance(.70):continue
            sid=self.series_add(name,'habit',cat,cycle,period)
            d=self.open+pd.Timedelta(days=self.integer(1,period))
            while d<=END:
                if cat=='wallet':
                    self.event(self.time(d),self.pick([100,150,200,300]),'habit',sid,txn_type='promptpay_out',channel='KPLUS',description='PromptPay To: EWALLET',**self.cp('WALLET01','EWALLET','004','ewallet_id'))
                else:self.payment(self.time(d),self.amount(amt,.10),sid,'qr',merchant,'5411' if cat=='groceries' else '7230')
                gap=self.integer(24,50) if cat=='personal_care' else self.integer(3,15) if cat=='wallet' else period
                d+=pd.Timedelta(days=gap)
        # v8: small bills paid to a person (a shared subscription split with a friend): a bill, monthly, fixed amount
        if self.chance(.35):
            amt=self.pick([35,41.5,59,70,99,119,149,199])
            sid=self.series_add('shared subscription (friend)','obligation','subscription','calendar',30.4,False)
            cpf=self.cp('SHARE01',self.pick(['นาย พีระ ม****','น.ส. ฝน ก****','นาย ต้น ส****']),self.pick(['004','014','002']),'mobile')
            ttype=self.pick(['promptpay_out','transfer_out'])
            for due in monthly(self.open+pd.Timedelta(days=self.integer(0,20)),self.integer(5,25)):
                if self.chance(.08):self.quirk(sid,'skipped_cycles');continue
                d=self.time(due+pd.Timedelta(days=self.integer(0,2)))
                self.event(d,amt,'obligation',sid,txn_type=ttype,channel='KPLUS',description=('PromptPay To: ' if ttype=='promptpay_out' else 'โอนเงิน โอนไป ')+cpf['counterparty_name'],**cpf)
        # v8: small irregular transfers to friends (food splits, paying back): NOT a bill
        if self.chance(.5):
            sid=self.series_add('small transfers to friends','habit','p2p','irregular',8)
            cpf=self.cp('FRIEND02',self.pick(['น.ส. แพร ช****','นาย บอส ท****','น.ส. มิ้นท์ ร****']),'004','mobile')
            d=self.open+pd.Timedelta(days=self.integer(1,10))
            while d<=END:
                self.event(self.time(d),self.amount(self.pick([45,60,90,120,180,250]),.35),'habit',sid,txn_type=self.pick(['promptpay_out','transfer_out']),channel='KPLUS',description='โอนเงิน โอนไป '+cpf['counterparty_name'],**cpf)
                d+=pd.Timedelta(days=self.integer(2,20))
        if self.chance(.3):
            sid=self.series_add('regular friend transfers','habit','p2p','interval',14)
            for d in pd.date_range(self.open+pd.Timedelta(days=2),END,freq='14D'):
                self.payment(self.time(d),self.pick([100,150,200]),sid,'p2p',cp=self.cp('FRIEND01','นาย กานต์ จ****','014','mobile'))

    def linked(self):
        ds=self.salary_dates
        semi=self.persona=='salaried_semimonthly'
        divisor=2 if semi else 1
        period=15.2 if semi else 30.4
        if self.chance(.60 if self.style=='family_supporter' else .42):
            sid=self.series_add('support parents','recurring_transfer','p2p','income_linked',period,True)
            for d in ds:self.payment(self.time(d+pd.Timedelta(days=1)),self.salary*.055/divisor,sid,'p2p',cp=self.cp('PARENT01','นาง มาลี ก****','014','mobile'))
        if self.chance(.28):
            sid=self.series_add('fund DCA','recurring_transfer','investment','income_linked',period)
            for d in ds:self.payment(self.time(d+pd.Timedelta(days=2)),round(self.salary*.025/divisor/100)*100,sid,'bill','FUND DCA',refkey=sid[-3:])
        if self.second or self.chance(.6):
            sid=self.series_add('transfer to own savings','own_transfer','savings','income_linked',period)
            for d in ds:
                date=self.time(d+pd.Timedelta(days=1))
                dest=self.second if self.second and date>=self.second_open else f'EXT{self.i:05d}'
                e=self.event(date,self.salary*.06/divisor,'own_transfer',sid,txn_type='transfer_out',channel='KPLUS',description='โอนเงินระหว่างบัญชี',counterparty_account_id=dest,counterparty_bank_code='004' if dest==self.second else '014',counterparty_name=self.name)
                if e:e['_own_dest']=dest
        if self.chance(.45):
            sid=self.series_add('shopping after payday','habit','shopping','income_linked',period)
            for d in ds:self.payment(self.time(d+pd.Timedelta(days=self.integer(1,3))),self.amount(500/divisor,.4),sid,'card','SHOPEE','5311')

    def everyday(self):
        # Frequency is independent of a target row count; insufficient cash removes
        # discretionary purchases rather than manufacturing funding/income.
        rates={'frugal':1.50,'foodie':2.0,'shopper':1.65,'commuter':1.9,'cafe_lover':2.1,'traveller':1.6,'family_supporter':1.6,'heavy_spender':2.3}
        merchants=[('7-ELEVEN','5411',65),('CAFE AMAZON','5814',65),('LINE MAN','5812',120),('GRAB','4121',105),('SHOPEE','5311',310),('LOCAL FOOD','5812',55),('RESTAURANT','5812',190),('BIG C','5411',180),('CINEMA','7832',240),('FRIEND','',120),('ATM','',500)]
        weights=np.array([.20,.13,.12,.09,.05,.20,.05,.04,.015,.075,.025])
        boost={'foodie':2,'shopper':4,'commuter':3,'cafe_lover':1,'traveller':3,'family_supporter':9,'heavy_spender':4}.get(self.style)
        if boost is not None:weights[boost]*=2
        weights/=weights.sum()
        for d in pd.date_range(self.open,END.normalize(),freq='D'):
            payday_near=any(0<=(d-p).days<=3 for p in self.salary_dates if abs((d-p).days)<5)
            lam=.78*rates[self.style]*(1.3 if payday_near else 1)*(1.1 if d.weekday()>=5 else 1)
            for _ in range(int(self.rng.poisson(lam))):
                j=int(self.rng.choice(len(merchants),p=weights));merchant,mcc,base=merchants[j]
                amt=self.amount(base,.40);dt=self.time(d)
                if merchant=='ATM':
                    amt=self.pick([500,1000,1500]);self.event(dt,amt,'habitual',txn_type='atm_withdrawal',channel='ATM',description='ATM CASH WITHDRAWAL')
                elif merchant=='FRIEND':
                    self.event(dt,amt,'habitual',txn_type='promptpay_out',channel='KPLUS',description='PromptPay To: เพื่อน',**self.cp('P'+str(self.integer(1,12)),'นาย ธนา น****','014','mobile'))
                else:
                    card=self.chance(.35)
                    self.event(dt,amt,'habitual',txn_type='card_purchase' if card else 'qr_payment',channel='CARD' if card else 'KPLUS',description=self.pick([merchant,merchant+' BKK',merchant+' TH']),merchant_id=merchant_code(merchant),merchant_name=merchant,mcc=mcc,original_currency='THB',original_amount=amt)
            if d.month in [4,12] and self.chance(.003):
                merchant=self.pick(['AGODA','AIRASIA']);amt=self.amount(2300,.35)
                self.event(self.time(d),amt,'one_off',txn_type='card_purchase',channel='CARD',description=merchant,merchant_id='M'+merchant,merchant_name=merchant,mcc='7011' if merchant=='AGODA' else '4511',original_currency='THB',original_amount=amt)
            if self.chance(.002):
                merchant=self.pick(['CLINIC','ELECTRONICS','CLOTHES']);amt=self.amount(1000 if merchant!='ELECTRONICS' else 6000,.3)
                self.event(self.time(d),amt,'one_off',txn_type='card_purchase',channel='CARD',description=merchant,merchant_id='M'+merchant,merchant_name=merchant,mcc='8011' if merchant=='CLINIC' else '5732' if merchant=='ELECTRONICS' else '5651',original_currency='THB',original_amount=amt)
            if self.chance(.003) and d+pd.Timedelta(days=3)<=END:
                dt=self.time(d);amt=self.pick([900,1200,1500])
                dinner=self.event(dt,amt,'one_off',txn_type='card_purchase',channel='CARD',description='RESTAURANT GROUP DINNER',merchant_id='MRESTAURANT',merchant_name='RESTAURANT',mcc='5812',original_currency='THB',original_amount=amt)
                for k in [1,2]:
                    self.event(dt+pd.Timedelta(days=k),amt/3,'reimbursement',flags='shared_meal',link=dinner['_token'],direction='CR',txn_type='promptpay_in',channel='KPLUS',description='PromptPay From: เพื่อน',**self.cp('DINNER'+str(k),'นาย กานต์ จ****','014','mobile'))

    def settle(self, offset):
        balance=dict(self.opening);external=self.external_opening
        output=[];retained=set();next_token=self.etoken
        initial=list(self.events);self.events=[]
        top_sid=''
        def synthetic(d, amt, account, direction, other, flag='', sid='', link=''):
            nonlocal next_token
            next_token+=1
            row={k:'' for k in TXCOLS}
            row.update(customer_id=self.cid,account_id=account,posted_datetime=d,amount=amt,direction=direction,txn_type='transfer_in' if direction=='CR' else 'transfer_out',channel='KPLUS',description='โอนเงินระหว่างบัญชี',counterparty_account_id=other,counterparty_bank_code='004' if other in balance else '014',counterparty_name=self.name)
            row.update(_kind='own_transfer',_sid=sid,_flags=flag,_link=link,_token=f'E{next_token:07d}',_own_dest='')
            return row
        def post(e):
            a=e['account_id'];balance[a]+=e['amount'] if e['direction']=='CR' else -e['amount']
            assert balance[a]>=0,(self.cid,a,'negative balance')
            e['balance_after']=balance[a]
            e['_order']=len(output)
            output.append(e);retained.add(e['_token'])
        for e in sorted(initial,key=lambda e:(e['posted_datetime'],e['_token'])):
            if e['_link'] and e['_link'] not in retained:
                self.skipped['dependent_on_skipped_purchase']+=1;continue
            a=e['account_id'];amt=e['amount']
            if e['direction']=='DR':
                mandatory=e['_kind'] in ['obligation','recurring_transfer']
                cushion=cents(max(600,self.salary*.14))
                if not mandatory and balance[a]-amt<cushion:
                    self.skipped['discretionary_or_savings_when_short']+=1;continue
                if mandatory and balance[a]<amt:
                    need=amt-balance[a]+cents(200)
                    if not top_sid:top_sid=self.series_add('own savings liquidity topups','own_transfer','savings','irregular',30.4)
                    if self.second and e['posted_datetime']>=self.second_open:
                        take=min(need,balance[self.second])
                        if take:
                            debit=synthetic(e['posted_datetime'],take,self.second,'DR',self.main,'',top_sid)
                            credit=synthetic(e['posted_datetime'],take,self.main,'CR',self.second,'topup_when_short',top_sid,debit['_token'])
                            post(debit);post(credit);need-=take
                    if need:
                        if external<need:raise ValueError(f'External savings exhausted for {self.cid}: need {need/100}, have {external/100}')
                        external-=need
                        post(synthetic(e['posted_datetime'],need,self.main,'CR',f'EXT{self.i:05d}','topup_when_short',top_sid))
            post(e)
            if e['_own_dest']:
                dest=e['_own_dest']
                if dest in balance:
                    post(synthetic(e['posted_datetime'],amt,dest,'CR',a,'',e['_sid'],e['_token']))
                else:external+=amt
        output.sort(key=lambda e:(e['account_id'],e['posted_datetime'],e['_order']))
        mapping={e['_token']:f'T{offset+j+1:010d}' for j,e in enumerate(output)}
        tx=[];labels=[]
        for e in output:
            row={k:e[k] for k in TXCOLS};row['txn_id']=mapping[e['_token']]
            row['posted_datetime']=e['posted_datetime'].strftime('%Y-%m-%d %H:%M:%S')
            row['amount']/=100;row['balance_after']/=100
            labels.append([row['txn_id'],e['_kind'],e['_sid'],e['_flags'],mapping[e['_link']] if e['_link'] else ''])
            tx.append(row)
            if e['_sid']:self.smap[e['_sid']]['observed_transactions']+=1
        for s in self.series:
            if s['cycle_type']=='irregular':
                dates=sorted(set(e['posted_datetime'].normalize() for e in output if e['_sid']==s['series_id']))
                if len(dates)>1:s['true_period_days']=round(float(np.mean(np.diff(np.array(dates,dtype='datetime64[D]')).astype(int))),3)
        self.external_final=external
        return pd.DataFrame(tx,columns=TXCOLS),pd.DataFrame(labels,columns=LBCOLS)

    def generate(self,offset):
        self.income();self.bills();self.intervals();self.linked();self.everyday()
        return self.settle(offset)

def gz_text(path):
    # mtime=0 and filename='' make gzip output byte-reproducible too.
    return io.TextIOWrapper(gzip.GzipFile(filename='',mode='wb',fileobj=open(path,'wb'),compresslevel=6,mtime=0),encoding='utf-8',newline='')

def write_csv(df,path):df.to_csv(path,index=False,encoding='utf-8',float_format='%.2f',lineterminator='\n')

def generate(users,seed,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    global_rng=np.random.default_rng(seed)
    nsemi=round(users*.10);nfree=round(users*.15)
    personas=['salaried']*(users-nsemi-nfree)+['salaried_semimonthly']*nsemi+['freelancer']*nfree
    global_rng.shuffle(personas)
    held=set(global_rng.choice(users,size=round(users*.15),replace=False).tolist())
    second=set(global_rng.choice(users,size=round(users*.4),replace=False).tolist())
    customers=[];accounts=[];series=[];utruth=[];total=0;skipped=collections.Counter()
    with gz_text(out/'transactions.csv.gz') as tf,gz_text(out/'txn_labels.csv.gz') as lf:
        for i in range(users):
            c=Customer(i+1,personas[i],i in second,i in held,seed)
            tx,lb=c.generate(total)
            tx.to_csv(tf,index=False,header=i==0,float_format='%.2f',lineterminator='\n')
            lb.to_csv(lf,index=False,header=i==0,lineterminator='\n')
            total+=len(tx)
            customers.append([c.cid,c.name,str(c.open.date())])
            accounts.extend([c.cid,a,'savings',str(d.date())] for a,d in c.account_open.items())
            series.extend(c.series);skipped.update(c.skipped)
            utruth.append(dict(customer_id=c.cid,user_split=c.split,persona=c.persona,base_monthly_salary=c.salary,housing=c.housing,spending_profile=c.style,history_class='short' if (END-c.open).days<180 else 'medium' if (END-c.open).days<365 else 'long',history_days=(END.normalize()-c.open).days+1,payday_rule=c.rule,new_payday_rule=c.new_rule,job_change_date=str(c.job_change.date()) if c.job_change is not None else '',annual_april_raise=c.raise_rate,opening_balances_json=json.dumps({a:b/100 for a,b in c.opening.items()},sort_keys=True),external_savings_opening=c.external_opening/100,external_savings_closing=c.external_final/100,validation_eligible=c.open<=pd.Timestamp('2026-06-01') or c.persona!='freelancer'))
            if (i+1)%100==0 or i+1==users:print(f'Generated {i+1}/{users} customers; {total:,} transactions',flush=True)
    write_csv(pd.DataFrame(customers,columns=CUCOLS),out/'customers.csv')
    write_csv(pd.DataFrame(accounts,columns=ACCOLS),out/'accounts.csv')
    write_csv(pd.DataFrame(series),out/'series_truth.csv')
    # Avoid rounding the generator's 3-5% raise parameter to only two decimals.
    pd.DataFrame(utruth).to_csv(out/'users_truth.csv',index=False,encoding='utf-8',lineterminator='\n')
    report=validate(out)
    report['seed']=seed;report['skipped_candidate_transactions']=dict(skipped)
    report['calendar_sources']=SRC
    report['assumptions']=[
      'Synthetic data: generation probabilities are scenario assumptions, not estimates of Thai customer prevalence.',
      'Amounts are THB; posted_datetime is naive local Asia/Bangkok time.',
      'The allowed global period starts 2024-01-01, but the 24-month opening rule places the earliest possible account opening at 2024-09-30. No earlier rows are fabricated.',
      'Nationwide BOT bank holidays are used for business-day pay rules; regional/Islamic-bank additions are excluded. The fixed 28th rule does not shift weekends or holidays.',
      'All salaried accounts open by 2026-04-01. Freelancers opening after 2026-06-01 have validation_eligible=False and cannot be evaluated as of that day.',
      'Opening balances are carried in users_truth.opening_balances_json, not injected as fake income. Infer the opening balance from the first transaction when using inputs only.',
      'Same-bank own-account transfers have paired debit/credit rows; external savings is a finite separately tracked pool, with opening and closing amounts in users_truth.',
      'Credit-card purchases model bank-account debits. Credit-card statement bills represent a separate external card, to avoid counting its purchases twice.',
      'Series status is status at 2026-09-30; start/end/successor information is truth only. Sparse annual series may have zero or one observed payment.',
      'Cancelled plans and skipped cycles create absent events; future ground truth must be computed from realized transaction labels within each validation window.',
      'Salary raises use a per-customer 3-5% April rate. The 2026 baseline has an THB 18k floor so earlier historical monthly salaries remain within THB 15k-100k after reversing raises and job changes.',
      'No fraud labels or fraud performance claims are implied by this recurring-bill dataset.'
    ]
    (out/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report

def validate(out):
    """Independently reload exported rows as strings, preserving bank codes."""
    out=Path(out)
    cu=pd.read_csv(out/'customers.csv',dtype=str,keep_default_na=False)
    ac=pd.read_csv(out/'accounts.csv',dtype=str,keep_default_na=False)
    sr=pd.read_csv(out/'series_truth.csv',dtype=str,keep_default_na=False)
    us=pd.read_csv(out/'users_truth.csv',dtype=str,keep_default_na=False)
    assert list(cu)==CUCOLS and list(ac)==ACCOLS
    assert cu.customer_id.is_unique and ac.account_id.is_unique and sr.series_id.is_unique and us.customer_id.is_unique
    assert set(cu.customer_id)==set(us.customer_id)==set(ac.customer_id)
    for frame in [cu,ac]:assert not any(c.startswith(('gt_','truth_')) or c in ['kind','series_id','labels'] for c in frame)
    owners=dict(zip(ac.account_id,ac.customer_id));opens=dict(zip(ac.account_id,ac.open_date));byseries=sr.set_index('series_id').to_dict('index');customer_names=dict(zip(cu.customer_id,cu.customer_name))
    balances={};external={};opening_total=0
    for r in us.to_dict('records'):
        for a,v in json.loads(r['opening_balances_json']).items():balances[a]=round(v*100);opening_total+=round(v*100)
        external[r['customer_id']]=round(float(r['external_savings_opening'])*100)
    first_balances=dict(balances);last_time={};all_ids=set();pending_links={};pay=collections.defaultdict(list);txkind=collections.Counter();flags=collections.Counter();statuses=collections.Counter(sr.status);counts=collections.Counter();obs=collections.Counter();own_pairs=collections.defaultdict(list)
    earliest=None;latest=None;total=0;minbal=None;usd=0;datesplit=collections.Counter();duplicate_ids=set();refund_links=set();split1=set();split2links=set()
    amount_sum=0;last_numeric=0
    tx_reader=pd.read_csv(out/'transactions.csv.gz',dtype=str,keep_default_na=False,chunksize=100000)
    lb_reader=pd.read_csv(out/'txn_labels.csv.gz',dtype=str,keep_default_na=False,chunksize=100000)
    for tx,lb in zip(tx_reader,lb_reader,strict=True):
        assert list(tx)==TXCOLS and list(lb)==LBCOLS
        assert len(tx)==len(lb) and tx.txn_id.tolist()==lb.txn_id.tolist()
        assert not any(c.startswith(('gt_','truth_')) or c in ['kind','series_id','labels'] for c in tx)
        for t,l in zip(tx.itertuples(index=False,name=None),lb.itertuples(index=False,name=None)):
            tid,cid,aid,dt,amount,direction,typ,channel,bal,desc,cp,bank,cpname,proxy,biller,ref1,ref2,mandate,mcc,mid,mname,cur,org=t
            _,kind,sid,fl,link=l
            numeric=int(tid[1:]);assert numeric==last_numeric+1;last_numeric=numeric
            assert tid not in all_ids;all_ids.add(tid)
            assert owners[aid]==cid and opens[aid]<=dt[:10]
            assert '2024-01-01 00:00:00'<=dt<='2026-09-30 23:59:59'
            assert aid not in last_time or dt>=last_time[aid];last_time[aid]=dt
            assert len(amount.split('.')[-1])==2 and len(bal.split('.')[-1])==2
            am=round(float(amount)*100);ba=round(float(bal)*100);assert am>0 and ba>=0
            assert direction in ['CR','DR'];balances[aid]+=am if direction=='CR' else -am
            assert balances[aid]==ba,(aid,tid,balances[aid],ba)
            assert channel in ['KPLUS','CARD','AUTO','ATM','BRANCH']
            assert typ in (['payroll_in','transfer_in','promptpay_in','refund_in'] if direction=='CR' else ['bill_payment','direct_debit','card_purchase','qr_payment','promptpay_out','transfer_out','atm_withdrawal'])
            assert not bank or (bank.isdigit() and len(bank)==3)
            assert not mcc or (mcc.isdigit() and len(mcc)==4)
            if typ in ['bill_payment','direct_debit']:
                assert biller and ref1 and not any([mid,mname,cp,bank,cpname,proxy])
                if typ=='direct_debit':assert mandate and channel=='AUTO'
            if typ in ['card_purchase','qr_payment']:
                assert all([mid,mname,mcc,cur,org]) and not any([cp,bank,cpname,proxy,biller,ref1,ref2,mandate])
                assert cur in ['THB','USD'] and float(org)>0
            if typ in ['transfer_in','transfer_out','promptpay_in','promptpay_out']:
                assert cp and bank and cpname and not any([mid,mname,biller,ref1,ref2,mandate])
                if typ.startswith('promptpay'):assert proxy in ['mobile','citizen_id','ewallet_id']
            if typ=='payroll_in':
                assert channel=='AUTO' and desc.startswith('PAYROLL ') and dt[11:19]<'06:00:00' and cp and bank and cpname
                pay[cid].append(dt[:10])
            if sid:assert sid in byseries and byseries[sid]['customer_id']==cid;obs[sid]+=1
            if sid and kind not in ['refund']:assert kind==byseries[sid]['kind']
            assert kind in ['obligation','recurring_transfer','income','own_transfer','habitual','one_off','habit','reimbursement','refund']
            if link:pending_links[tid]=(link,cid,am,fl,kind,dt)
            if kind=='own_transfer':
                if cp in owners:
                    assert owners[cp]==cid and bank=='004'
                    pairkey=(cid,dt,am,tuple(sorted([aid,cp])))
                    own_pairs[pairkey].append(direction)
                else:
                    assert cpname==customer_names[cid]
                    external[cid]+=am if direction=='DR' else -am
                    assert external[cid]>=0
            tags=fl.split('|') if fl else []
            flags.update(tags)
            if 'duplicate_charge' in tags:duplicate_ids.add(tid)
            if kind=='refund':refund_links.add(link)
            if 'split_part1' in tags:split1.add(tid)
            if 'split_part2' in tags:split2links.add(link)
            if 'topup_when_short' in tags:assert kind=='own_transfer' and direction=='CR' and typ=='transfer_in'
            txkind[kind]+=1;counts[cid]+=1
            earliest=dt if earliest is None else min(earliest,dt);latest=dt if latest is None else max(latest,dt)
            datesplit['train_dates' if dt<'2025-10-01' else 'calibration_dates' if dt<'2026-02-01' else 'test_dates']+=1
            minbal=ba if minbal is None else min(minbal,ba);usd+=int(cur=='USD');total+=1
    assert all(link[0] in all_ids for link in pending_links.values())
    assert all(sorted(v)==['CR','DR'] for v in own_pairs.values())
    assert duplicate_ids<=refund_links and split2links<=split1 and split1<=split2links
    assert all(obs[sid]==int(s['observed_transactions']) for sid,s in byseries.items())
    for r in us.to_dict('records'):assert external[r['customer_id']]==round(float(r['external_savings_closing'])*100)
    # Salary checks use standard input fields only, including bonus credits.
    payday=[];gaps=collections.defaultdict(list)
    for r in us.to_dict('records'):
        if r['persona']=='freelancer':continue
        ds=sorted(pd.Timestamp(d) for d in pay[r['customer_id']]);june=[d for d in ds if d.year==2026 and d.month==6]
        d=june[0] if june else None;previous=[x for x in ds if d is not None and x<d];nxt=[x for x in ds if d is not None and x>d]
        gap=(nxt[0]-d).days if nxt else None
        estimated=float(np.median(np.diff(np.array(previous[-5:]+([d] if d is not None else []),dtype='datetime64[D]')).astype(int))) if len(previous)>=2 else 30.0
        ok=[bool(june),len(previous)>=2,gap is not None and 10<=gap<=40,10<=estimated<=40]
        payday.append(ok)
        if gap is not None:gaps[r['persona']].append(gap)
    arr=np.array(payday,dtype=float)
    assert arr.size==0 or bool((arr==1).all()),'Payday eligibility failed'
    assert latest[:10]=='2026-09-30'
    stats={k:round(float(v),2) for k,v in pd.to_numeric(us.base_monthly_salary).describe(percentiles=[.1,.5,.9]).to_dict().items()}
    cross=sr.groupby(['kind','cycle_type']).size().reset_index(name='series').to_dict('records')
    return dict(checks_passed=True,customers=len(cu),accounts=len(ac),transactions=total,transaction_labels=total,recurring_series=len(sr),heldout_customers=int((us.user_split=='heldout').sum()),heldout_share=float((us.user_split=='heldout').mean()),second_account_share=float((ac.groupby('customer_id').size()==2).mean()),first_transaction=earliest,last_transaction=latest,minimum_balance=minbal/100,same_bank_own_transfer_pairs=len(own_pairs),usd_transactions=usd,transaction_counts_by_kind=dict(txkind),flags=dict(flags),series_statuses=dict(statuses),series_by_kind_and_cycle=cross,baseline_monthly_salary_thb=stats,history_days=pd.to_numeric(us.history_days).describe().to_dict(),rows_by_date_split=dict(datesplit),payday_check=dict(salaried_customers=len(payday),june_salary_share=float(arr[:,0].mean()) if arr.size else None,two_prior_credits_share=float(arr[:,1].mean()) if arr.size else None,next_credit_10_to_40_days_share=float(arr[:,2].mean()) if arr.size else None,estimated_gap_10_to_40_days_share=float(arr[:,3].mean()) if arr.size else None,median_next_gap_days={k:float(np.median(v)) for k,v in gaps.items()}),validation_ineligible_late_freelancers=int((us.validation_eligible=='False').sum()),checks=['exact input schemas and field order','unique IDs and customer/account ownership','positive amounts and exact cent-level balance reconciliation','nonnegative internal and finite external savings balances','opening dates and per-account chronology','one-to-one transaction labels and valid series/links','no truth columns in model input files','leading-zero string code formats and payment-specific fields','paired same-bank own transfers','duplicate refunds and two-part bill links','series observation counts and label kinds','June and subsequent payday eligibility','history extends through 2026-09-30'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--users',type=int,default=2500)
    parser.add_argument('--seed',type=int,default=20261013)
    parser.add_argument('--out',default='data_v8')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if args.users<1 or args.seed<0:parser.error('--users must be positive and --seed nonnegative')
    if args.validate_only:print(json.dumps(validate(args.out),ensure_ascii=False,indent=2))
    else:generate(args.users,args.seed,args.out)

if __name__=='__main__':main()
