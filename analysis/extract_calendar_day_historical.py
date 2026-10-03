"""Historical 1981-2020 fixed-calendar-day AMS; same algorithm as future period."""
import traceback
import extract_calendar_day_future as pipeline

pipeline.YEARS=range(1981,2021)
pipeline.OUT=pipeline.ROOT/'ukcp18_historical_calendar_day_ams_gev'

if __name__=='__main__':
    try:
        pipeline.main()
    except Exception:
        pipeline.OUT.mkdir(exist_ok=True)
        (pipeline.OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
