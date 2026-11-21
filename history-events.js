/* Display-only event context from the published regional report. No catalog acquisition or scoring. */
(function(root,factory){
    const api=factory();
    if(typeof module==='object'&&module.exports) module.exports=api;
    else root.GeoSpecHistoryEvents=api;
})(typeof globalThis==='object'?globalThis:this,function(){
    'use strict';
    const PREFIX='retained-event:';
    const escape=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    function eventTime(value){
        if(typeof value!=='string') return null;
        const match=value.match(/^(\d{4}-\d{2}-\d{2})(?:T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?$/);
        if(!match) return null;
        const date=Date.parse(match[1]+'T00:00:00Z');
        if(!Number.isFinite(date)||new Date(date).toISOString().slice(0,10)!==match[1]) return null;
        if(!match[2]) return {day:match[1],utc_time:null};
        if(+match[2]>23||+match[3]>59||+match[4]>59) return null;
        // earthquake_events.py emits naive ISO strings from datetime.utcfromtimestamp.
        const time=Date.parse(value+(/(?:Z|[+-]\d{2}:\d{2})$/.test(value)?'':'Z'));
        if(!Number.isFinite(time)) return null;
        const utc=new Date(time).toISOString();
        return {day:utc.slice(0,10),utc_time:utc};
    }
    function selectRegion(payload,region,days){
        const entry=payload?.[region];
        if(!entry||typeof entry!=='object'||(entry.region&&entry.region!==region))
            return {status:'UNAVAILABLE',events:[],retained:0,outside:0,invalid:0,conflicts:0,count:null,updated:null};
        const sources=[...(Array.isArray(entry.events)?entry.events:[]),entry.largest_event,entry.most_recent_event,
            ...(Array.isArray(entry.m65_plus_events)?entry.m65_plus_events:[])].filter(Boolean);
        const byId=new Map(),conflicts=new Set();let invalid=0;
        for(const record of sources){
            const time=eventTime(record.time),id=record.event_id;
            if(!time||typeof id!=='string'||!id.trim()||typeof record.magnitude!=='number'||!Number.isFinite(record.magnitude)) {invalid++;continue;}
            const event={...time,id,region,magnitude:record.magnitude,place:typeof record.place==='string'?record.place:'Location not retained',
                source_time:record.time,url:typeof record.url==='string'&&/^https:\/\/earthquake\.usgs\.gov\/earthquakes\/eventpage\/[a-zA-Z0-9_-]+\/?$/.test(record.url)?record.url:null};
            const previous=byId.get(id);
            if(previous&&(previous.day!==event.day||previous.magnitude!==event.magnitude||
                (previous.utc_time&&event.utc_time&&previous.utc_time!==event.utc_time))) conflicts.add(id);
            if(!previous||(!previous.utc_time&&event.utc_time)) byId.set(id,event);
        }
        const all=[...byId.values()].filter(e=>!conflicts.has(e.id)).sort((a,b)=>a.day.localeCompare(b.day)||String(a.utc_time).localeCompare(String(b.utc_time))||a.id.localeCompare(b.id));
        const dates=days.map(d=>d.date).filter(d=>eventTime(d)?.day===d).sort();
        const events=all.filter(e=>dates.length&&e.day>=dates[0]&&e.day<=dates[dates.length-1]);
        return {status:'READY',events,retained:all.length,outside:all.length-events.length,invalid,conflicts:conflicts.size,
            count:Number.isInteger(entry.event_count)&&entry.event_count>=0?entry.event_count:null,updated:entry.last_updated||null};
    }
    function describe(e){
        return 'M'+e.magnitude+' | '+e.place+' | '+e.region+'\n'+
            (e.utc_time?'Event time (UTC): '+e.utc_time:'Event day (UTC): '+e.day+'; exact time not retained')+
            '\nEvent ID: '+e.id+'\nVertical line: UTC calendar day. Same-day order versus report-day values is unknown; not a prediction or causal match.';
    }
    function overlays(events,selectedId,enabled=true){
        const shapes=[],annotations=[];
        if(!enabled) return {shapes,annotations};
        const groups=new Map();events.forEach(e=>{if(!groups.has(e.day))groups.set(e.day,[]);groups.get(e.day).push(e);});
        for(const [day,group] of groups){
            const selected=group.find(e=>e.id===selectedId);
            shapes.push({name:PREFIX+day,type:'line',xref:'x',yref:'paper',x0:day,x1:day,y0:0,y1:1,
                line:{color:'#fb7185',width:selected?2:1,dash:'dashdot'}});
            annotations.push({name:PREFIX+day,xref:'x',yref:'paper',x:day,y:1,xanchor:'right',yanchor:'bottom',showarrow:false,
                text:selected?'▼ M'+selected.magnitude+(group.length>1?' ('+group.length+' events)':''):'▼',
                hovertext:group.map(e=>escape(describe(e)).replace(/\n/g,'<br>')).join('<br><br>'),
                font:{color:'#fb7185',size:10},bgcolor:'rgba(17,17,27,0.8)'});
        }
        return {shapes,annotations};
    }
    return {PREFIX,eventTime,selectRegion,describe,overlays};
});
