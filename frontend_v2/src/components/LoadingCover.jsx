import styles from './LoadingCover.module.css'

export default function LoadingCover() {
  return (
    <div className={styles.root}>
      <div className={styles.inner}>
        <span className={styles.tag}>STATE OF THE ART SURVEY</span>
        <h1 className={styles.title}>The Autonomous<br/>Mind.</h1>
        <p className={styles.sub}>Loading synthesis…</p>
        <div className={styles.bar}><div className={styles.fill} /></div>
      </div>
    </div>
  )
}
